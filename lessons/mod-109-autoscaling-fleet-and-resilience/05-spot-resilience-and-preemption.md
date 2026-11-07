# Chapter 05 — Spot Resilience: Preemption Notice, Drain, Checkpointed KV, Cross-Region Failover

> A GPU spot instance costs roughly a third of its on-demand price. The catch is a short preemption notice (tens of seconds on major clouds) and no SLA on when the next spot slot becomes available. Naive spot adoption turns every preemption into an SLO breach; the right discipline turns it into a cheap-and-reliable backbone. This chapter builds that discipline — the preemption signal, the drain protocol, the handoff, and when KV checkpointing is worth the complexity vs. simply re-routing.

## Motivation

A platform team wants to cut their inference bill. On-demand H100s dominate the budget. They adopt spot for a reactive-pool slice:

- Chapter 04's warm pool stays on-demand (critical: never let the warm pool preempt).
- The reactive scale-out replicas run on spot H100s.
- They calculate: spot is ~1/3 the price; even with occasional preemption the net is a 60 % budget cut on the reactive layer.

Day one: a spot preemption fires at 14:03. The pod gets a `SIGTERM` with 60 seconds notice (AWS). What happens next depends entirely on what the pod does with that 60 seconds:

- **Baseline bad case:** The pod ignores the signal. 60 seconds later, `SIGKILL`. The 12 in-flight requests get `connection reset`. The client retries; the retries land on a reactive-pool replica that is *also* being preempted because the spot capacity is tight across the whole region. At 14:05, 30 requests have 503'd. TTFT P99 is in the tens of seconds because the retry queue is scrambling for warm replicas.
- **Partial good case:** The pod catches `SIGTERM`, returns `503 Service Unavailable` to new requests, keeps the in-flight ones running. The in-flight requests complete at 30+ seconds of generation; they make it before `SIGKILL`. New arrivals get 503s but the retry at the client succeeds on another replica. SLO is bruised but not broken.
- **Full good case:** The pod catches `SIGTERM`, stops admitting, finishes the in-flight requests, broadcasts "I'm going" to the load balancer *before* the readiness probe flips, and the LB stops routing to it. By `SIGKILL` the pod is already out of rotation and empty. The SLO is unaffected.

The chapter is the discipline that turns the partial-good into the full-good. It covers:

- The preemption signal on each major cloud (AWS, GCP, Azure) and the time budget it gives.
- The drain protocol — stop admitting, finish in-flight, de-register from the LB, exit.
- When KV checkpointing pays and when it doesn't — a specific cost/benefit, not a default.
- Cross-region failover for when a whole region loses spot capacity simultaneously.
- Provider-fallback at the edge (chapter 06 formalises).

By the end of this chapter you can:

- Wire the cloud's spot preemption signal into a pod lifecycle handler.
- Author the drain controller — a sidecar or in-engine handler that stops admitting, finishes in-flight, and de-registers.
- Reason about KV checkpointing — the engine's support, the HBM-to-host-RAM path, the request-restart semantics, and the specific workloads where it pays.
- Design failover routing that catches the request a drained replica would have served.
- Compose with chapter 04's warm pool and chapter 06's multi-provider routing.

Cloud preemption-notice windows per current docs: AWS EC2 Spot is a 2-minute warning (<https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html>, with no documented shorter window for GPU instance types); GCP Spot/preemptible is best-effort up to 30 seconds, with an opt-in 120-second notice available (<https://cloud.google.com/compute/docs/instances/spot>); Azure Spot is best-effort 30 seconds via Scheduled Events (<https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms>). Always re-verify against current cloud docs for the specific instance type.

## The spot preemption signal

Each cloud emits a preemption signal through its instance metadata service (IMDS). The pod's lifecycle hook has from "notice arrives" to "instance terminates" to drain.

### AWS — Spot interruption notice

- Signal path: `/latest/meta-data/spot/instance-action` via IMDSv2. Returns a JSON with `action` and `time` (the termination moment).
- Window: AWS documents a **2-minute** warning for EC2 Spot Instance interruption with no shorter window called out for GPU instance types (see <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html>). Design for the documented window but treat it as best-effort — planning for a conservative sub-window (e.g. ~60 seconds of usable drain) leaves headroom.
- Spot rebalance recommendation (longer-lead, less-certain): `/latest/meta-data/events/recommendations/rebalance` — an earlier signal that AWS *may* reclaim the instance, useful for speculative drain.

Reference: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html>.

### GCP — Preemptible / Spot VM

- Signal path: `metadata.google.internal` → `instance/preempted` or the ACPI shutdown signal (SIGTERM).
- Window: Google documents a **30-second** notice for Spot VMs and traditional preemptible VMs.
- Compute Engine also emits a maintenance event signal for live-migration (not preemption, but a similar drain opportunity).

Reference: <https://cloud.google.com/compute/docs/instances/spot>.

### Azure — Spot VM

- Signal path: Scheduled Events endpoint `http://169.254.169.254/metadata/scheduledevents?api-version=...`.
- Window: Azure Spot VMs have a **30-second** notice via Scheduled Events.

Reference: <https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms>.

### Kubernetes node-termination handlers

Each cloud has an open-source handler that watches the IMDS and taints the node / cordons it / triggers pod drain:

- **AWS** — `aws-node-termination-handler` (<https://github.com/aws/aws-node-termination-handler>).
- **GCP** — GKE's node-problem-detector handles preemption; standalone controllers also exist.
- **Azure** — the `kured` family or custom controllers using Scheduled Events.

Install one of these on the GPU node pool that runs spot. When a preemption arrives, the handler cordons the node and sends `SIGTERM` to pods; the pod's `preStop` hook and / or signal handler executes the drain. (`aws-node-termination-handler` ships two mutually-exclusive modes — an IMDS Processor DaemonSet that polls IMDS for the spot ITN, scheduled events, rebalance recommendations, and ASG lifecycle state, and a Queue Processor Deployment consuming an SQS queue fed by EventBridge; toggle via `enableSqsTerminationDraining`. For GPU workloads the IMDS Processor with the rebalance signal is the common shape — see <https://github.com/aws/aws-node-termination-handler>.)

## The drain protocol

A drain is a four-step contract between the preempting pod, the LB, and the rest of the fleet:

### Step 1 — Stop admitting

The pod flips a flag: "I'm draining; `readinessProbe` returns 503." The LB's health check sees unready; the LB stops routing new requests to this pod. Existing TCP connections (SSE streams for in-flight generations) continue.

Implementation: a `preStop` hook or SIGTERM handler sets an internal flag that the readiness-probe path reads:

```yaml
spec:
  containers:
  - name: vllm
    # ...
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "touch /var/run/draining && sleep 55"]
    readinessProbe:
      httpGet: {path: /ready, port: 8000}
      periodSeconds: 2
      failureThreshold: 1
```

The `/ready` endpoint in the engine checks for `/var/run/draining` and returns 503 if present. The `sleep 55` gives the drain time to complete (adjust for your preemption window).

For engines without a `/ready` endpoint that honours a drain flag, a sidecar container exposes one, and the Deployment's `Service` points at the sidecar's port.

### Step 2 — Finish in-flight

Let the batched generation complete for the requests already running. For an SSE-streaming inference request, this means continuing to write tokens to the TCP connection until the generation ends (or the max-tokens budget caps it).

Operational wrinkle: an LLM generation can take **long** — tens of seconds for a long response. The preemption window may not accommodate all generations. In practice:

- **Cap `max-tokens` or `max-generation-time` during drain.** New requests already can't admit; existing requests might be told to wrap up at the next sentence (engine-specific feature).
- **Abort cleanly past the budget.** If the drain window ends before the longest in-flight finishes, return a graceful `{"error": "server draining"}` or trailer rather than connection-reset. The client's retry will pick up another replica.

### Step 3 — De-register from the LB

Modern service meshes / ingress controllers remove unready pods from the backend pool. If you're on plain Kubernetes Services + kube-proxy, the service endpoint is removed when readiness flips. Verify in your setup that the time from "readiness flips" to "no new connections arrive" is single-digit seconds; if it's longer, investigate the LB.

### Step 4 — Exit

Once in-flight is drained and the LB has stopped sending, the process can exit cleanly. The container terminates; the pod is removed; the preemption-handler acknowledges to the cloud (where applicable) that the node can be reclaimed.

### A full example — SIGTERM handler for a vLLM-shape server

Rough shape for a wrapper around `vllm serve` (engines typically have built-in support — this is the fallback if the engine doesn't):

```python
import signal, time, threading, os

draining = threading.Event()

def handle_sigterm(signum, frame):
    print("SIGTERM received; entering drain", flush=True)
    draining.set()                      # flips readiness
    deadline = time.time() + 50          # 10s buffer below AWS 60s
    while time.time() < deadline:
        if engine.num_in_flight() == 0:
            print("drained; exiting", flush=True)
            os._exit(0)
        time.sleep(1)
    print("drain window elapsed; aborting remaining requests", flush=True)
    engine.abort_all(status="server draining")
    os._exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
```

Both vLLM and SGLang expose equivalents — vLLM via a `--shutdown-timeout` flag that aborts in-flight requests and drains output queues on SIGTERM/SIGINT (see RFC <https://github.com/vllm-project/vllm/issues/63734> for current contract and known gaps, including the `--api-server-count > 1` bug), and SGLang via its native SIGTERM drain in the Python server (`TokenizerManager.sigterm_watchdog`) plus the Rust server's drain-on-SIGTERM path. The TRT-LLM / Triton stack inherits Triton's graceful-shutdown behaviour. Verify your engine's drain interface against the pinned release's docs; the pattern above is a fallback when the engine has no built-in support.

## KV checkpointing — when it pays, when it doesn't

The drain protocol loses the KV state of in-flight requests when the pod exits — the next replica can't resume mid-generation, so if the drain timed out and aborted a request, the client's retry re-pays prefill entirely. For long-prefill / short-generation workloads, that's expensive.

**KV checkpointing** writes an in-flight request's KV state to host RAM (or shared storage) so another replica can resume. Three flavours:

- **Host-RAM checkpointing within the node** — copy HBM → host RAM; another pod on the same node (or next pod on the node) can read it back. Fast (PCIe bandwidth). Doesn't survive node termination — useless on spot preemption.
- **Shared-tier checkpointing (remote KV cache)** — write KV to NVMe or an object store; any replica anywhere can read it. Slow (bandwidth-limited); significant engine work. See LMCache / Mooncake / NIXL in chapter 104.
- **Full request snapshot, re-prefill elsewhere** — don't actually save KV; record the prompt and the partially-generated response, send both to another replica, let it prefill the whole concatenation and continue. Simpler engine-side; re-pays prefill.

The economics:

```
Does the SLO tolerate request re-prefill on preemption?
├── Yes → No checkpointing; drain + client retry.
└── No  → Continue.

Is the prefill large (say, >4K tokens) and the drain budget short?
├── Yes → Remote KV cache (LMCache / Mooncake / NIXL) if your engine supports it.
└── No  → Full snapshot re-prefill; simpler and robust.

Is the fleet's spot preemption rate high (e.g. >5% per hour)?
├── Yes → Checkpointing investment pays off faster.
└── No  → Drain + retry is often enough; keep it simple.
```

For most fleets, **the simplest shape wins**: aggressive drain + aggressive retry with chapter 06's provider-fallback. Checkpointing is worth the complexity when:

1. The workload has long prefills (RAG with 16K+ tokens).
2. The SLO is tight (no tolerance for client-visible retries).
3. The preemption rate is high enough that you measurably lose requests to it.

Chapter 104's tiered-KV-store discussion is the technical depth; this chapter's responsibility is naming whether the investment is worth it. (Coverage across the three remote-KV projects is uneven as of writing: LMCache — <https://github.com/LMCache/LMCache> — has an explicit vLLM integration but no documented SGLang path; Mooncake — <https://github.com/kvcache-ai/Mooncake> — documents integrations with both vLLM and SGLang; NVIDIA's NIXL — <https://github.com/ai-dynamo/nixl> — is a transfer-engine abstraction targeting the NVIDIA Dynamo stack and doesn't directly integrate with vLLM/SGLang. "KV survives pod restart" is not universal; verify engine-side support before adopting.)

## Spot pool shape and placement

Three fleet-shape decisions matter:

### Decision 1 — which workloads are spot-safe

- **On-demand only:**
  - Chapter 04's **warm pool** — never spot. A preempting warm pool defeats the whole pattern.
  - **Long-prefill / stateful / tenant-isolated** workloads where a retry is expensive.
  - **Draft-target** replicas during traffic where the draft's state must not be lost mid-generation.

- **Spot OK:**
  - Chapter 04's **reactive scale-out pool** — spot is the whole point. These replicas come and go anyway.
  - **Stateless batch** — generation that retries are cheap.
  - **Embedding / rerank / small-model** workloads.

The split is a per-workload policy (chapter 06 formalises).

### Decision 2 — spot diversity

A single spot pool on one instance type in one zone correlates its preemptions — AWS reclaims p5.48xlarge capacity in us-east-1a; your whole spot fleet empties at once. Diversity mitigations:

- **Multiple instance types.** `p5.48xlarge`, `p5e.48xlarge`, `p4d.24xlarge` all GPU-capable; the Karpenter `NodePool` can list several. A capacity shortage on one is independent of the others.
- **Multiple availability zones.** Spread spot across zones; a zonal capacity event affects only its zone.
- **Multiple regions.** The last-resort diversity. Chapter 06 covers multi-region patterns.

### Decision 3 — on-demand fallback for spot pool

Configure the node autoscaler to fall back to on-demand if spot is unavailable:

```yaml
# Karpenter NodePool
spec:
  template:
    spec:
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]       # prefer spot; fall back to on-demand
```

Karpenter's spot-to-on-demand fallback is the robust shape; Cluster Autoscaler can be configured similarly via multiple node groups.

## Cross-region and cross-provider failover

When a whole region loses spot capacity (or a cloud event), the local resilience budget is exhausted. Two layers of failover:

### Layer 1 — same cloud, different region

Deploy replicas in `us-east-1` and `us-west-2` behind a weighted-DNS or global LB (chapter 06). Baseline weighting sends most traffic to the closest region; on a regional capacity event, the LB's health check failures route traffic to the other region.

The win: the SLO survives a one-region spot event.

The cost: cross-region data transfer is not free; the "other region" needs enough on-demand capacity to absorb failover traffic (a cold-start pile-up on the backup side is itself an SLO breach).

### Layer 2 — different cloud or external provider

A LiteLLM-shape router (chapter 06) can route to an external provider (Fireworks, Together, Baseten, etc.) when self-hosted capacity is exhausted. The win: the fleet's resilience is bounded by the sum of self-hosted + provider capacity, not either alone.

The costs: data egress to the provider, possibly worse per-token latency, tenant-data-boundary considerations (chapter 06 flags), and per-provider rate limits.

## Composing with chapter 04

Chapter 04's warm pool is the primary defense against cold-start; chapter 05's drain is the primary defense against preemption. They compose:

- **Warm pool absorbs preemption-driven churn.** When a reactive-spot replica preempts, the warm pool catches the traffic until the next spot replica boots.
- **Reactive scaler sees the preemption as scale-out signal.** The preempting replica's traffic shifts to the warm pool; chapter 02's HPA sees the warm pool saturating and triggers scale-out of a new spot replica.
- **Drain prevents the retry storm.** If the pod drains cleanly, the clients' retries succeed on the warm pool; if it doesn't, clients see `connection reset` and retries pile on simultaneously.

Chapter 05's bet is that you can afford to lose a reactive replica now and then, provided:

1. Chapter 04's warm pool is sized to catch the loss without breaking the SLO.
2. The drain protocol prevents retry storms by letting in-flight requests complete and new requests re-route gracefully.
3. The spot pool is diversified enough that simultaneous preemption of the whole reactive layer is unlikely.

## Alerts around spot resilience

Four alerts the spot pattern needs:

### Alert 1 — drain violations

A pod was `SIGKILL`'d before `SIGTERM`'s grace period expired. Something is wrong — either the drain protocol isn't working, or the `terminationGracePeriodSeconds` is too short for the drain window.

```promql
# Count SIGKILL events (Kubernetes annotates terminated pods; node-termination-handler events)
sum(rate(kube_pod_terminated_total{reason="Preempted"}[5m]))
```

### Alert 2 — in-flight drops during drain

If the engine exposes in-flight counters, drain should drain them to zero before exit. If a replica terminates with `num_in_flight > 0`, some requests were killed mid-stream.

```promql
# vllm-specific; adjust per engine
max_over_time(vllm_num_requests_running{pod="draining-pod"}[30s]) > 0
  and on(pod) kube_pod_container_status_terminated{pod="draining-pod"}
```

### Alert 3 — spot fleet drain rate

Chapter 06 and the SRE discipline (module 110) want to know the drain rate. Sustained > 10 % preemption / hour is a signal the spot mix needs more diversity, or the fleet should shift more capacity to on-demand.

```promql
rate(aws_spot_interruption_total[1h]) * 3600
```

### Alert 4 — simultaneous drains

If > 20 % of the spot fleet is draining in the same 60 seconds, the warm pool can't absorb it all. Page early; manually scale the on-demand reactive layer up.

## Common deployment errors

### Error 1 — `terminationGracePeriodSeconds` too short

Default is 30 s; AWS gives ~60 s of warning. The pod has 30 s to drain and is `SIGKILL`'d by Kubernetes before the drain completes. Fix:

```yaml
spec:
  terminationGracePeriodSeconds: 90   # > cloud preemption window + buffer
```

### Error 2 — no preStop or signal handler

The pod ignores `SIGTERM`; in-flight requests get `connection reset`. Fix: implement the drain contract.

### Error 3 — readiness probe honours drain flag with a delay

The engine sets the drain flag but the readiness-probe poll interval is 10 seconds; the LB keeps sending requests for 10 extra seconds. Fix: tighter `periodSeconds` on readiness (2–5 s) and `failureThreshold: 1`.

### Error 4 — the warm pool is on spot

Scenario: platform team sets everything to spot to maximise savings. A regional spot event simultaneously preempts the warm pool and the reactive pool. SLO breaks; cold-start hits every subsequent retry. Fix: warm pool on on-demand always.

### Error 5 — spot without diversity

One instance type, one AZ. A capacity event removes the whole fleet. Fix: diversify instance types; across AZs; (where regional events are a concern) multi-region.

### Error 6 — client retries without backoff

Even with a clean drain, aggressive retries (no backoff, no jitter) from clients create a thundering-herd on the surviving replicas. Make sure the client library applies exponential backoff; the ingress / proxy can rate-limit retries if the clients don't.

### Error 7 — KV checkpointing on every request

Checkpointing is cheap at low volume and expensive at scale. Enabling it for every request to "be safe" burns PCIe bandwidth and HBM. Fix: checkpoint only when preemption notice is actually imminent; the engine's drain handler is the trigger.

### Error 8 — forgetting cross-region failover tested

A region failover plan that has never been tested is a hope, not a plan. Chapter 06 includes the drill; this chapter flags it.

## The boundary to the level-20 peer and chapter 06

This chapter does not teach:

- **Cloud spot pricing** or the business-case math for spot adoption. That's team-lead-track (L40) / cross-functional-leadership territory.
- **Generic preemption handling** — the chapter uses GPU-inference specifics; the mechanism of catching SIGTERM is level-20 generic.
- **Multi-region load balancing at the DNS / Anycast layer.** Chapter 06 of this module covers that; the chapter here only integrates to it.

This chapter *does* teach: the drain protocol for an inference replica, the KV-checkpointing cost/benefit, the pool-shape discipline, and the alert set for a spot-aware fleet.

## Summary

- Spot GPU is ~1/3 the on-demand price with ~30–60 s preemption notice (cloud-specific). Naive adoption creates SLO breaches; the drain protocol turns preemption into controlled churn.
- The drain protocol: **stop admitting (readiness flip) → finish in-flight → de-register from LB → exit**. All within `terminationGracePeriodSeconds` set above the cloud's preemption window.
- KV checkpointing is a cost/benefit: pays only for long-prefill / tight-SLO / high-preemption-rate workloads. Most fleets win with drain + client-retry.
- Fleet shape: **warm pool on-demand (never spot), reactive scale-out on spot with diversity (multi-instance-type, multi-AZ, cross-region fallback).**
- Alert on `terminationGracePeriod` violations, in-flight > 0 at terminate, sustained spot preemption rate, and simultaneous-drain-of-fleet.
- Compose with chapter 04 (warm pool absorbs preemption-driven churn) and chapter 06 (cross-region / cross-provider failover is the next ring of resilience).
- Common errors (short grace period, no signal handler, warm pool on spot, no diversity, naive retries) all preventable with the chapter's discipline.
- Generic Kubernetes graceful-shutdown and cloud spot fundamentals route to the level-20 peer; this chapter owns the inference-specific drain and the fleet-shape decisions.

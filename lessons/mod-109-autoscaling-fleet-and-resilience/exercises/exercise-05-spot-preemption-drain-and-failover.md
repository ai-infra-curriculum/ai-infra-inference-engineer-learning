# Exercise 05 — Spot Preemption, Drain, and Failover

**Time:** 4 hours · **Output:** one spot-backed reactive node pool, one preemption-handler install, one drain-protocol wiring (readiness flip + in-flight finish + LB de-register + exit), one injected-preemption drill with observed SLO, one spot-pool-shape decision doc, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 05](../05-spot-resilience-and-preemption.md) concrete by **standing up a spot-backed reactive pool**, wiring the preemption-handler and the drain protocol, injecting a preemption against a running replica, and proving the SLO survives the drain. The deliverable is the drain contract the fleet runs on — the thing that turns a preemption event from an incident into a controlled rollover.

## Problem statement

The reactive tier from exercise 04 currently runs on on-demand. Spot H100s are a fraction of the price with a tens-of-seconds preemption notice; the chapter 05 opening motivates the move. The exercise is the discipline that makes it safe: the preemption signal is caught, the drain happens inside the window, the LB stops routing before the kill, and the warm pool from exercise 04 absorbs the gap. Run the drill, prove the behaviour, and write down the pool-shape policy before you run it in production.

## Prerequisites

- Chapter 05 of this module read. Chapters 01, 02, and 04 skimmed (substrate; scaler re-absorption; warm-pool composition).
- **Exercises 01, 02, 04 complete** — GPU Operator, Prometheus, HPA/KEDA scaler, and a two-tier Deployment with warm + reactive pools. The exercise moves the reactive tier onto spot.
- Cluster on a cloud with spot / preemptible capability: AWS (Spot), GCP (Spot / preemptible), or Azure (Spot VM). The cloud's node-termination-handler is installable with `cluster-admin`.
- A representative spot instance type for your GPU class — `p5.48xlarge` spot on AWS, `a3-highgpu-8g` spot on GCP, `NC_A100_v4` spot on Azure, etc. Verify spot availability in your target region before planning.
- A way to inject or simulate a preemption. Current options:
  - AWS: `aws fis` (Fault Injection Simulator) EC2-spot-interruption action — real preemption against a specific instance. Verify current FIS action name in the AWS docs; the API has evolved.
  - AWS (secondary): the node-termination-handler's test-mode or an IMDSv2 mock sidecar.
  - GCP: `gcloud compute instances simulate-maintenance-event` or manually stopping the instance.
  - Azure: Scheduled Events simulation via the metadata endpoint, or a controlled VM stop.
  - Portable fallback: a direct `kubectl delete pod <reactive-pod>` with the right `terminationGracePeriodSeconds` set — simulates `SIGTERM` but not the IMDS signal path.
- Load driver from exercises 02 / 04 (same burst script; use it to put real traffic on the pod during the preemption drill).

## Requirements

Six deliverables.

### Deliverable A — the spot-backed reactive pool

Stand up a GPU node pool on spot capacity. Shape depends on your cloud:

**AWS / Karpenter:** update exercise 01's Karpenter `NodePool` (or create a new one) for the reactive tier with mixed spot + on-demand fallback and instance-type diversity:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-h100-spot
spec:
  template:
    spec:
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["p5.48xlarge", "p5e.48xlarge", "p4d.24xlarge"]   # diversity
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]                              # spot preferred; on-demand fallback
      - key: topology.kubernetes.io/zone
        operator: In
        values: ["us-east-1a", "us-east-1b", "us-east-1c"]         # AZ diversity
      taints:
      - key: inference-tier
        value: reactive-spot
        effect: NoSchedule
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-h100
```

(Cluster Autoscaler users: one node group per instance type across AZs, tagged for spot, with on-demand node groups as fallback; chapter 01 shape.)

Update exercise 04's reactive Deployment to tolerate `inference-tier=reactive-spot:NoSchedule` so it lands on this pool; leave the warm pool's toleration pointed at the on-demand pool (chapter 05 Error 4 is non-negotiable — the warm pool never runs on spot).

Attach:

- The `NodePool` / node-group manifest.
- The updated reactive Deployment with the new toleration.
- Evidence the reactive pods landed on spot nodes: `kubectl get nodes -L karpenter.sh/capacity-type` showing spot capacity; `kubectl get pods -o wide` showing reactive pods on those nodes.
- Confirmation the warm pool pods did NOT land on spot (`kubectl get pods -o wide` for the warm Deployment on on-demand only).

### Deliverable B — the node-termination handler

Install the cloud's node-termination handler on the spot node pool:

- **AWS:** `aws-node-termination-handler` (<https://github.com/aws/aws-node-termination-handler>). Pick the IMDS Processor DaemonSet mode or the Queue Processor Deployment mode per chapter 05; for a GPU reactive pool, the IMDS mode with rebalance-recommendation + spot ITN enabled is the common shape.
- **GCP:** GKE's node-problem-detector covers spot preemption natively; verify it's enabled on the GPU node pool. Or install a standalone handler if your distribution doesn't.
- **Azure:** `kured` or a Scheduled-Events-watching controller; verify your cluster's shape.

Required config knobs regardless of cloud:

- Watch for the spot preemption signal (ITN on AWS, preemption notice on GCP, scheduled event on Azure).
- Cordon and drain the node on signal.
- `terminationGracePeriodSeconds` on the drained pods respected (chapter 05 Error 1).
- A node-level taint applied during drain so new pods don't schedule onto the doomed node.
- For AWS specifically: optionally also watch the earlier rebalance recommendation signal.

Attach:

- The handler install (helm command, values overrides).
- `kubectl -n <ns> get pods` showing the handler running on every spot node.
- The handler's metric endpoint (or logs) showing it's watching the IMDS / scheduled-events.

### Deliverable C — the drain-protocol wiring

Implement chapter 05's four-step drain contract on the reactive pods:

1. **Stop admitting** — the engine flips to "draining" on `SIGTERM`. Readiness probe returns 503. For a vLLM / SGLang / TRT-LLM pod, most current releases ship a built-in `SIGTERM` drain (vLLM's `--shutdown-timeout` and SGLang's `TokenizerManager.sigterm_watchdog`); if your engine release doesn't, add chapter 05's `preStop` hook and the fallback Python wrapper. Verify against the current release's drain contract (vLLM RFC: <https://github.com/vllm-project/vllm/issues/63734>).
2. **Finish in-flight** — the engine lets batched generations complete to their `max_tokens` budget or sentence-boundary cap; new requests are rejected.
3. **De-register from LB** — the Service endpoint is removed on readiness-flip; verify the time from readiness-flip to endpoint-removal is single-digit seconds.
4. **Exit** — `os._exit(0)` after in-flight drains, or the drain budget elapses and the engine aborts remaining requests with a graceful error.

Required manifest additions on the reactive Deployment:

```yaml
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 90       # > cloud preemption window + buffer
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

If your engine doesn't natively honour `/var/run/draining`, wire a sidecar that reads the file and overrides the readiness endpoint (chapter 05 names the pattern).

Attach:

- The reactive Deployment with `terminationGracePeriodSeconds`, `lifecycle.preStop`, and the tightened `readinessProbe`.
- The engine's drain configuration (e.g. `--shutdown-timeout=50`).
- A simple test: `kubectl delete pod <reactive-pod>` and watch:
  - The pod enters `Terminating`.
  - `kubectl get endpoints <service>` loses the pod's IP within seconds.
  - The pod stays around for up to `terminationGracePeriodSeconds` while in-flight generations complete.
  - The pod exits cleanly without a `SIGKILL`.

### Deliverable D — the injected-preemption drill

Run the real drill with traffic on the pod. Script:

1. **Baseline** — route a sustained load from exercise 02's load driver through the Service. Confirm TTFT P99 on the dashboard is at steady state (under SLO); confirm at least 3–5 reactive pods are in rotation.
2. **Inject** — using the cloud's test API (AWS FIS, GCP maintenance-event, Azure scheduled-events sim) target one reactive pod's underlying node. If you don't have the test API, `kubectl delete pod <reactive-pod>` with the full drain config in place is a fair second-best (document the fallback).
3. **Observe** during the 60–120 seconds that follow:
   - **Readiness flip** — the targeted pod's readiness goes to 503 within a few seconds.
   - **Endpoint removal** — the Service endpoints lose the pod's IP within a few seconds of the flip.
   - **In-flight completion** — the engine's `num_requests_running` on the draining pod goes to zero before `terminationGracePeriodSeconds` elapses (chapter 05 Alert 2 inverse — on the good path, we want zero remaining).
   - **Warm-pool absorption** — TTFT P99 on the Service stays under SLO; the warm pool (and other reactive pods) catch the rerouted traffic.
   - **Clean exit** — the pod terminates without an `OOMKilled` / `SIGKILL` on the container; events show `Terminated: Completed`.
   - **New reactive pod** — the HPA from exercise 02 sees the capacity dip, scales up a replacement reactive pod (chapter 05 composition with chapter 02).

Record:

```
# Preemption Drill — <date>

## Setup
- Target deployment: <name>
- Target pod: <name>, on node <node_name>, SKU <SKU>, capacity-type: spot
- Baseline load: <RPS> requests/s, <N> warm + <M> reactive replicas
- Injection method: <AWS FIS | kubectl delete | GCP maintenance | ...>

## Timeline
t+0.0 s     Preemption signal injected
t+<a> s     Readiness flipped to 503 (from pod logs)
t+<b> s     Service endpoints lost the pod's IP (`kubectl get endpoints`)
t+<c> s     In-flight requests reached zero on the draining pod
t+<d> s     Pod exited cleanly; container `Terminated: Completed`
t+<e> s     HPA desired replicas increased (scale-up triggered)
t+<f> s     New reactive pod Running + Ready

## SLO observation
- TTFT P99 during drill: <value> (SLO: <target>) — PASS/FAIL
- 5xx rate during drill: <value> (SLO: <target>) — PASS/FAIL
- Any client-visible retries? <yes / no — if yes, diagnose>
```

Attach the dashboard screenshot (TTFT P99, queue depth, in-flight, replica count) across the drill window.

### Deliverable E — the pool-shape decision document

A one-page policy that names the fleet's spot-pool shape:

```
# Spot Pool Shape — <fleet_name>

## On-demand only (NEVER spot)
- Chapter-04 warm pool (chapter 05 Error 4).
- Long-prefill / stateful / tenant-isolated workloads (chapter 05 Decision 1).
- Draft-target speculative replicas where draft state loss is unacceptable.

## Spot OK (reactive tier)
- Reactive scale-out pods behind the HPA.
- Stateless batch and embedding workloads.
- Rerank and small-model workloads with cheap-retry contracts.

## Diversity policy (chapter 05 Decision 2)
- Instance types: at least 2 compatible SKUs per pool (e.g. p5 + p4d).
- AZs: at least 2 within the region.
- Capacity-type fallback: spot preferred, on-demand fallback enabled.

## Drain budget
- terminationGracePeriodSeconds: <seconds, > cloud preemption window>
- Engine shutdown timeout: <seconds, less than the K8s grace period>
- Buffer: ≥10 seconds between engine abort and K8s SIGKILL.

## KV checkpointing decision
- Flavour: <none | host-RAM (on-node) | shared-tier (LMCache/Mooncake) | full snapshot re-prefill>
- Reasoning: <SLO tolerance, prefill length, preemption rate — chapter 05 "KV checkpointing — when it pays" matrix>

## Failover
- Same-cloud multi-region: <yes / no; chapter 06 exercise>
- Multi-provider: <yes / no; chapter 06 exercise>

## Alerts (from Deliverable F)
- Drain violations (SIGKILL before grace elapsed)
- In-flight > 0 at pod terminate
- Sustained preemption rate > 10% / hour
- Simultaneous drain > 20% of spot fleet in 60s

## Review date
- Monthly: preemption rate per instance type / AZ; adjust diversity.
- On incident: any SLO breach traced to a preemption triggers review.
```

### Deliverable F — the alert pack + `<!-- needs-escalation -->` list

Implement chapter 05's four alerts as `PrometheusRule` entries:

1. **Drain violations** — pods SIGKILL'd before `terminationGracePeriodSeconds` elapsed (via node-termination-handler events or kube-state-metrics).
2. **In-flight drops during drain** — `num_requests_running > 0` at pod terminate.
3. **Spot fleet drain rate** — sustained > 10 % / hour preemption; the specific PromQL shape in chapter 05 Alert 3 is cloud-specific (Karpenter emits its own events; `aws-node-termination-handler` has metrics).
4. **Simultaneous drain** — > 20 % of the spot fleet in a 60-second window.

Attach the `PrometheusRule` manifest and one simulated firing (easiest: temporarily lower the drain rate threshold and trigger a controlled kill).

Then produce `<!-- needs-escalation -->`:

- "The right KV-checkpointing flavour for our prefill-heavy RAG class. Chapter 05 flags LMCache / Mooncake / NIXL coverage as uneven; engine support and operational experience are both limited." → chapter 104 remote-KV discussion; vendor docs for our pinned engine.
- "The cost model on spot interruption — how many preemptions per hour does our SLO budget tolerate?" → module 110 error-budget methodology; chapter 114 budget-vs-SLA.
- "AWS FIS (or GCP equivalent) permissions in our security posture. Running a preemption injection may need platform-security sign-off." → cloud security; platform team.
- "Whether to wire the AWS rebalance-recommendation signal (longer lead, less certain) in addition to the ITN. Chapter 05 names it; the trade-off is speculative drain overhead vs. less panic." → `aws-node-termination-handler` docs; our error-budget allowance.
- "Cross-region failover cost. Exercise 06 adds the next ring; until then, a regional spot event is unmitigated." → exercise 06.

## Starter guidance

- **Keep the warm pool on on-demand. Non-negotiable.** Chapter 05 Error 4 is the one error that undoes the chapter's whole framing. If your Karpenter / autoscaler config is mixed-capacity by default, verify explicitly.
- **Verify the drain contract before the drill.** Deliverable C's simple `kubectl delete pod` test is a dress rehearsal. If `kubectl delete pod` with the full drain config still reset client connections mid-stream, the FIS drill will too. Fix at the dress-rehearsal layer first.
- **Match the grace period to the cloud window.** AWS Spot ITN = 2 min, GCP Spot = 30 s, Azure Spot = 30 s. `terminationGracePeriodSeconds: 90` is right for AWS; 60 would be too tight. Shortening for a different cloud is fine; lengthening below the window is not.
- **The drill is the acceptance test, not an afterthought.** Deliverable D is where the chapter's claims become evidence. Without the injected preemption, Deliverables A / B / C are hopeful configuration. Make time for the drill.
- **Fall back to `kubectl delete pod` if the test API is blocked.** The IMDS signal path is cloud-specific; a plain SIGTERM via `kubectl delete pod` tests steps 1–4 of the drain protocol. Note the test-path difference (no node-cordon side effect) in Deliverable D.
- **Don't test on a live tenant.** Even with the drain contract working, the drill will produce *some* retries; the dashboard measurement is honest. Run on a staging deployment or a shielded tenant.
- **The dashboard is the audit trail.** Capture the TTFT / 5xx / in-flight / replica-count series for the drill window. The time-aligned view is what proves the SLO survived.

## Fallbacks for constrained environments

- **No cloud spot access.** The `kubectl delete pod` path exercises steps 1–4 of the drain; you miss the IMDS-signal + node-cordon integration but still prove the engine drain. Note the gap.
- **No node-termination-handler available for your cloud.** Write a simple handler as a controller watching the IMDS endpoint and tainting the node; this is a lot of work and chapter 05's intent is to install an existing one. Prefer upgrading the cluster or moving the exercise to a cloud that supports one.
- **Engine has no SIGTERM drain.** Build the Python wrapper from chapter 05's example. Verify the engine's `/health` endpoint can be made conditional (it may be hardcoded in some releases).
- **Can't install a `PrometheusRule`.** Author the alert expressions in a doc and mark them as "to-wire" when the Prometheus install stack allows. Deliverable F's expressions are what you need; the manifest is a wrapping.
- **Preemption rate is zero on your test cluster.** You can't observe real spot preemptions; the drill (Deliverable D) is the substitute. Note in Deliverable F that the drain-rate alert is pre-emptive; it will fire at production scale.

## Acceptance criteria

1. **Deliverable A** shows reactive pods landing on a spot node pool with instance-type and AZ diversity; the warm pool is on on-demand only.
2. **Deliverable B** shows the cloud's node-termination handler installed on every spot node with the right signal watchlist.
3. **Deliverable C** shows the four-step drain contract (readiness flip + in-flight finish + LB de-register + clean exit) working under a controlled `kubectl delete pod` test, with `terminationGracePeriodSeconds` set above the cloud preemption window.
4. **Deliverable D** shows a real (or best-available-simulation) preemption drill with the timeline recorded; the SLO was observed and `PASS/FAIL` recorded; the warm pool absorbed the gap.
5. **Deliverable E** is a one-page policy that names the on-demand-only and spot-OK workloads, the diversity policy, the drain budget, the KV-checkpointing decision, and the alert set.
6. **Deliverable F** has the four-alert `PrometheusRule` manifest and a demonstrated firing; the `<!-- needs-escalation -->` list names at least five items.
7. The drill is reproducible — the same injection method, the same deployment config, and the same load profile should produce the same timeline and SLO observation.

## Common failure modes

- **`terminationGracePeriodSeconds: 30` (default) on a 60–120 s preemption window.** Chapter 05 Error 1. The pod is SIGKILL'd before drain completes; in-flight requests reset. Fix: set above the cloud window with a buffer.
- **No SIGTERM handler; engine ignores the signal.** Chapter 05 Error 2. Current vLLM / SGLang releases handle it natively; TRT-LLM / Triton inherits Triton's shutdown. If your pinned release doesn't, wire the wrapper.
- **Readiness probe at 10 s period.** Chapter 05 Error 3. The LB keeps routing for an extra 10 s after readiness flips. Fix: `periodSeconds: 2`, `failureThreshold: 1`.
- **Warm pool on spot.** Chapter 05 Error 4. The whole pattern breaks. Verify at Deliverable A.
- **Spot without diversity.** Chapter 05 Error 5. One instance type in one AZ; a capacity event removes the whole fleet. Verify the diversity policy in Deliverable E.
- **Client without backoff.** Chapter 05 Error 6. Retries slam the warm pool during the drain window. If you control the client, set exponential backoff with jitter; if you don't, rate-limit retries at the ingress.
- **KV-checkpointing on everything by default.** Chapter 05 Error 7. The cost is real; the win is workload-specific. Deliverable E's matrix is the defensive check.
- **Cross-region failover untested.** Chapter 05 Error 8 and chapter 06 — a regional-event-killing-all-spot scenario will overwhelm the local resilience. Note as `<!-- needs-escalation -->` if exercise 06 isn't run yet.
- **Drill without traffic.** A preemption drill on a pod with no in-flight requests doesn't prove the drain; it proves the engine can exit. Load the pod before injecting.
- **Dashboard not screen-capped.** The drill's evidence evaporates if no one snapshots the TTFT / 5xx series. Capture at drill time.

## Keep it

The spot-backed reactive pool (Deliverable A), the node-termination handler (Deliverable B), the drain contract on the reactive Deployment (Deliverable C), and the injected-preemption drill record (Deliverable D) become the fleet's resilience artefacts. The pool-shape policy (Deliverable E) is the document the on-call reads during a sustained preemption event. Exercise 06's multi-region shape uses this exercise's drain protocol per region; cross-region failover kicks in when the local reactive tier drains faster than it refills. The alert pack (Deliverable F) composes with module 110's SLO methodology — drain-rate alerts feed error-budget decisions that back the spot-vs-on-demand mix.

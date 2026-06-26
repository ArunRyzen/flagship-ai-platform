# System-Design Interview Q&A — for this platform

The capstone is a great whiteboard prompt: "design a production AI assistant." Here's how this
codebase answers it.

---

### Q. Design a production RAG + agent assistant. Walk me through the components.
Orchestrator (`Assistant`) → input **guardrails** → a **ReAct agent** whose tools include **hybrid
retrieval** → grounded, cited **answer**. Around it: a **semantic cache**, **rate limiting**,
**metrics**, **tracing**, and an **eval ship-gate** in CI. The LLM is one component; the other ~80%
is retrieval quality, control flow, safety, measurability, observability, and serving.

### Q. Where do guardrails go and why first?
At the very front, before the agent or any tool runs. Indirect prompt injection rides in the input
(or retrieved content), so you sanitize and block *before* the model can act on it. A blocked request
short-circuits with zero agent steps.

### Q. Why is retrieval a tool the agent calls, instead of a fixed pre-step?
Because the agent should *decide* when it needs to look something up — some questions need retrieval,
some need a calculation, some neither. Making `knowledge_search` a tool gives the model that control
(the agentic shape) while still grounding answers when it does retrieve.

### Q. How do you keep the answer grounded and attributable?
The retrieval tool records exactly which chunks it returned; those become the answer's citations, and
the system prompt instructs the model to answer only from retrieved context or say it doesn't know.

### Q. How do you stop the agent from running forever / blowing the budget?
A **step budget** caps the ReAct loop. Combined with rate limiting at the API and (in production) a
token/task budget, that bounds cost per request and per client.

### Q. How do you know a change didn't regress quality?
An **eval ship-gate**: `run_eval` scores the whole `Assistant` over a versioned golden set; `gate()`
fails CI below threshold. It runs as a CI step, so a prompt/model/retrieval change that drops quality
turns the build red.

### Q. How would you scale this to real traffic?
Swap in pgvector (shared state) and run multiple async workers behind a load balancer; back the cache
and rate limiter with Redis so they're shared across replicas; tier models by cost; and for
self-hosted inference, vLLM (continuous batching + KV cache) on GPUs orchestrated by Kubernetes.

### Q. What does observability look like here?
A per-request **trace** of nested spans (`ask → guardrails → agent`) plus a `/metrics` endpoint
(request count, cache hit rate). In production you'd export the spans to Langfuse / Phoenix / OTel and
alert on latency, error rate, cost, and drift.

### Q. What are the main failure modes and mitigations?
Injection (guardrails), retrieval miss (hybrid + rerank, better chunking), hallucination beyond
context (answer-only-from-context + faithfulness evals), runaway loops (step budget), and cost spikes
(caching + rate limits + model tiering). Each maps to a component in this design.

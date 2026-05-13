+++
draft = false
title = 'What I learned building a multi-model grammar benchmark'
+++

I've been building a framework to benchmark language models on grammatical error correction - comparing GPT-5.5, Claude 4.7 Opus, Flan-T5, CoEdit, and a few others on the BEA-2019 dataset. The core idea is simple: throw sentences with grammar errors at different models and measure which ones fix them best.

The implementation ended up being way more interesting than I expected. Not because of the models, but because of everything *around* the models.

## The architecture

The project is structured around a clean abstraction: each model provider implements a `BaseBackend` with a single `correct(text) -> str` method. A `BaseStrategy` wraps the prompt engineering - seq2seq, few-shot, edit-based - and an `Evaluator` orchestrates the pipeline: load dataset, run corrections, compute metrics.

The config-driven approach lets me swap models without touching code:

```yaml
models:
  gpt4o:
    provider: openai
    model: gpt-5.5
    temperature: 0.0
    max_tokens: 512

  claude-opus:
    provider: anthropic
    model: claude-opus-4-7
    temperature: 0.0
    reasoning: true

  t5-small:
    provider: huggingface
    model: google/flan-t5-small

  coedit-large:
    provider: huggingface
    model: grammarly/coedit-large
```

Run `llm-grammar-bench --config configs/default.yaml run --models t5-small,coedit-large` and it runs the full pipeline. Clean.

## The wall

Then I actually tried running it at scale.

The BEA-2019 validation set has thousands of sentences. Running each one through a separate API call means:

- OpenAI batch endpoints work fine - send 50K prompts, come back later, collect results. Not fast, but reliable.
- Every other provider expects you to hit them one-at-a-time with rate limits, retries, and exponential backoff.

The project has a clean retry decorator with configurable backoff:

```python
@retry(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
def _call_api(self, system_prompt, user_text, **overrides):
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

And a thread-pool executor with token-bucket rate limiting for concurrent calls:

```python
class BatchExecutor:
    def __init__(self, max_workers=5, rate_limiter=None):
        self._max_workers = max_workers
        self._rate_limiter = rate_limiter

    def map(self, func, items):
        # Concurrent execution with per-item rate limiting
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_wrap, item, idx): idx for idx, item in enumerate(items)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        return results
```

But here's the thing - I didn't write this infrastructure because I wanted to. I wrote it because every provider has a slightly different API surface, slightly different error formats, slightly different rate limit headers. I spent more time today building *the bridge* than *running the actual benchmark*.

## The providers

The project supports OpenAI, Anthropic, HuggingFace transformers, and OpenAI-compatible endpoints (OpenRouter, Groq, vLLM, OpenCode Go). Each has its own quirks:

- **OpenAI**: Clean API, batch endpoints exist, rate limits are predictable. The gold standard.
- **Anthropic**: No batch. Different error format. Different auth. Their model is strong but using it at scale means you're writing bespoke infrastructure.
- **HuggingFace (local)**: No rate limits, no networking, no auth. Just load the model and go. The models are smaller but the *experiment is reproducible* - same weights, same behavior, every time.
- **OpenAI-compatible** (Groq, OpenRouter, vLLM): They speak the OpenAI protocol but none have batch. Same single-call-at-a-time problem.

## The pivot

I started moving the heavy lifting to local models via the HuggingFace backend. The tradeoff is obvious - Flan-T5 Small isn't going to beat GPT-5.5 on grammar correction. But for what I'm actually measuring (comparative performance across error types), consistency matters more than raw capability.

With local models:
- No rate limit mystery - every call takes the same time
- No "our provider updated their model and your baseline is broken"
- No 3 AM retry loops because someone's CDN hiccupped
- No API key expiry mid-run

The caching layer handles the rest - same input gets the same correction, instantly:

```python
cached = self._cache.get(self.model_id, text)
if cached is not None:
    return cached

correction = self._call_api(system_prompt, text)
self._cache.set(self.model_id, text, correction)
return correction
```

## The CEFR insight

One thing that genuinely surprised me: the BEA-2019 dataset annotates errors by CEFR level (A1-C2). The evaluator breaks down performance by proficiency level automatically:

```
CEFR breakdown:
  A1: errant=0.7234
  B1: errant=0.5812
  C1: errant=0.4123
```

The pattern is consistent across every model I've tested - higher proficiency errors are harder to correct. This isn't surprising in retrospect (advanced grammar mistakes are subtler) but seeing it quantified across models is satisfying.

## Where this is going

The whole point of this benchmark is to pick the right backend for a writing assistant I'm building. Grammar correction is one slice of it - I also want to test style rewriting, tone adjustment, and summarization.

The cost-latency-quality tradeoff is the real question. A local model like CoEdit is free and fast but weaker. GPT-5.5 is strong but costs per-call and needs batch infrastructure. Something like OpenCode Go or Groq sits in the middle - cheaper than OpenAI, no batch, but fast enough that single calls don't hurt as much.

The framework makes it easy to answer that question with real numbers instead of guessing. Run the same dataset across every option, compare the scores, compare the runtime, compare the cost. Pick the winner for each task.

## What I'd change

If I were doing this again, I'd think harder about the abstraction layer upfront. The `BaseBackend` interface is clean but it hides the fact that each provider has fundamentally different scaling characteristics. A better approach might be a two-tier system:

1. **Sync API** - single corrections, low latency, for interactive use
2. **Batch API** - bulk corrections with callbacks or polling, for benchmarking

Only OpenAI genuinely supports both today. For everyone else, "batch" means "write your own retry loop and hope the process doesn't die at 4 AM."

## Bottom line

Building a cross-provider benchmark framework taught me more about API infrastructure than about grammar correction. The models themselves are fascinating, but the real engineering challenge is getting them to talk to you on your terms, not theirs.

If you're building evaluation tools, plan for the infrastructure overhead from day one. And consider local models - they're weaker but they're *honest*. No batching surprises, no midnight retries, just predictable results you can trust.

---

*The project is open source at [github.com/pepijnfrenken/grammar-benchmark](https://github.com/pepijnfrenken/grammar-benchmark) if you want to poke around or run your own benchmarks.*

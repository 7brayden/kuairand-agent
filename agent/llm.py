"""THE single LLM call site. No other module may talk to a model provider.

Token accounting is 15% of the score and depends on this being the only door:
every call — including failures and retries — records usage, so the journal's
per-iteration ``tokens_in``/``tokens_out`` sums are auditable against the
provider's own totals.

Prompts are versioned ``.md`` files under ``agent/prompts/`` (e.g. ``policy_v1.md``),
referenced by name — never inline strings in Python.

The provider SDK is imported lazily so the harness (and milestone-0 runs with the
stub generator) works with no API key and no ``anthropic`` install.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class LLMError(RuntimeError):
    """Provider call failed after exhausting retries. Never crashes the loop —
    the caller turns this into an ``llm_api_error`` ErrorEvent."""


@dataclass
class LLMResponse:
    """One completed call: text plus exact usage."""

    text: str
    tokens_in: int
    tokens_out: int
    prompt_name: str
    model: str


@dataclass
class LLMClient:
    """Provider wrapper. Construct exactly one per run.

    ``tokens_in``/``tokens_out`` accumulate across ALL calls in the current
    iteration; the loop drains them with :meth:`take_iteration_usage` when writing
    the journal entry, so retries and calls inside rejected iterations are counted
    rather than lost.
    """

    provider: str = "anthropic"   # "anthropic" | "azure" (Azure AI Foundry / Azure OpenAI)
    model: str = "claude-sonnet-5"   # for azure this is the DEPLOYMENT name, not a model id
    prompts_dir: Path = Path("agent/prompts")
    max_output_tokens: int = 8192
    effort: Optional[str] = None   # default output_config.effort; per-call override in complete()
    temperature: float = 1.0       # AZURE ONLY. Claude 5 models reject sampling params (400).
    max_retries: int = 3
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    _prompt_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _client: Any = field(default=None, repr=False)

    def load_prompt(self, prompt_name: str) -> str:
        """Read ``{prompts_dir}/{prompt_name}.md``. Versioning lives in the filename."""
        if prompt_name not in self._prompt_cache:
            path = Path(self.prompts_dir) / f"{prompt_name}.md"
            self._prompt_cache[prompt_name] = path.read_text(encoding="utf-8")
        return self._prompt_cache[prompt_name]

    def render(self, prompt_name: str, variables: dict[str, Any]) -> str:
        """Fill ``{placeholders}`` in the named prompt.

        Uses explicit replacement rather than ``str.format`` because prompts contain
        code samples full of braces that ``format`` would choke on.
        """
        text = self.load_prompt(prompt_name)
        for key, value in variables.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    def _ensure_client(self) -> Any:
        """Construct the provider client. Lazy, so no SDK or key is needed to import."""
        if self._client is not None:
            return self._client
        if self.provider == "azure":
            import os
            try:
                from openai import AzureOpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise LLMError(f"openai SDK not installed (pip install openai): {exc}") from exc
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not endpoint or not key:
                raise LLMError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
            self._client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        else:
            try:
                import anthropic  # imported lazily: milestone 0 needs no provider
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise LLMError(f"anthropic SDK not installed: {exc}") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def _call_azure(self, client: Any, prompt: str,
                    system: Optional[str]) -> tuple[str, int, int, str]:
        """One Azure OpenAI chat completion. Returns (text, tokens_in, tokens_out, finish).

        Differences from the Anthropic path that matter here: usage fields are named
        prompt_tokens/completion_tokens, truncation shows up as finish_reason == "length"
        rather than stop_reason == "max_tokens", and temperature is supported (Claude 5
        rejects it) — so it is set, since a little sampling diversity helps a weaker model
        avoid proposing the same hypothesis every iteration.
        """
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model=self.model,                 # Azure routes on DEPLOYMENT name
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        usage = resp.usage
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
                resp.choices[0].finish_reason or "")

    def complete(self, prompt_name: str, variables: dict[str, Any],
                 system: Optional[str] = None,
                 effort: Optional[str] = None) -> LLMResponse:
        """Render the named prompt and call the provider, with retry + backoff.

        Usage is recorded on EVERY attempt, including ones that then fail, because
        a failed call still costs input tokens. Raises :class:`LLMError` only after
        exhausting retries; the loop converts that into a journaled error event
        rather than a crash.
        """
        prompt = self.render(prompt_name, variables)
        client = self._ensure_client()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.calls += 1

                if self.provider == "azure":
                    text, t_in, t_out, finish = self._call_azure(client, prompt, system)
                    self.tokens_in += t_in
                    self.tokens_out += t_out
                    if finish == "length":
                        raise LLMError(
                            f"response truncated at max_tokens={self.max_output_tokens} "
                            f"(used {t_out} completion tokens). Raise llm.max_output_tokens.")
                    if finish == "content_filter":
                        raise LLMError("Azure content filter blocked the response")
                    return LLMResponse(text=text, tokens_in=t_in, tokens_out=t_out,
                                       prompt_name=prompt_name, model=self.model)

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_output_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # NOTE: temperature/top_p/top_k are REMOVED on Claude Opus 5 and Sonnet 5
                # and return a 400. Do not reintroduce them. Depth is controlled by
                # `effort` instead, and thinking is adaptive by default on Opus 5.
                chosen_effort = effort or self.effort
                if chosen_effort:
                    kwargs["output_config"] = {"effort": chosen_effort}
                if system:
                    kwargs["system"] = system
                msg = client.messages.create(**kwargs)
                self.tokens_in += msg.usage.input_tokens
                self.tokens_out += msg.usage.output_tokens

                # Check stop_reason BEFORE reading content. Both of these return HTTP 200
                # and would otherwise surface downstream as an unparseable response, which
                # sends the debug action chasing a code bug that does not exist.
                if msg.stop_reason == "max_tokens":
                    raise LLMError(
                        f"response truncated at max_tokens={self.max_output_tokens} "
                        f"(used {msg.usage.output_tokens} output tokens). Thinking counts "
                        f"toward output on Opus 5 — raise llm.max_output_tokens or lower "
                        f"llm.effort.")
                if msg.stop_reason == "refusal":
                    cat = getattr(getattr(msg, "stop_details", None), "category", None)
                    raise LLMError(f"model declined the request (category={cat})")

                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                return LLMResponse(text=text, tokens_in=msg.usage.input_tokens,
                                   tokens_out=msg.usage.output_tokens,
                                   prompt_name=prompt_name, model=self.model)
            except Exception as exc:  # provider errors are heterogeneous; treat alike
                last_exc = exc
                usage = getattr(exc, "usage", None)
                if usage is not None:  # a failure that still consumed tokens
                    self.tokens_in += getattr(usage, "input_tokens", 0)
                    self.tokens_out += getattr(usage, "output_tokens", 0)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))

        raise LLMError(f"{prompt_name}: provider failed after {self.max_retries} attempts: "
                       f"{last_exc}")

    def take_iteration_usage(self) -> tuple[int, int]:
        """Return (tokens_in, tokens_out) accumulated since the last take, and reset."""
        ti, to = self.tokens_in, self.tokens_out
        self.tokens_in = self.tokens_out = 0
        return ti, to


# ------------------------------ response parsing -------------------------------------
# Models wrap their output in prose no matter how firmly the prompt asks them not to,
# so we parse rather than trust. A parse failure is a `bad_llm_output` error event, not
# a crash — the caller reroutes.

def extract_code_block(text: str) -> str:
    """Pull the first fenced python block out of a response."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        raise LLMError("no fenced python block in response")
    return m.group(1)


def extract_json_block(text: str) -> dict:
    """Pull the first fenced json object out of a response.

    Falls back to the first bare ``{...}`` span, because models occasionally drop the
    fence; that fallback is worth more than a wasted iteration.
    """
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        raise LLMError("no json object in response")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"malformed json in response: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"expected a json object, got {type(data).__name__}")
    return data

from __future__ import annotations
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from app.config import settings
from app.logging_setup import get_logger
from app.utils.token_budget import enforce_token_budget

log = get_logger("openai_llm")
client = OpenAI(api_key=settings.openai_api_key)

class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str

def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    pricing = {
        "gpt-5": (settings.price_in_gpt5, settings.price_out_gpt5),
        "gpt-5-mini": (settings.price_in_gpt5_mini, settings.price_out_gpt5_mini),
        "gpt-5-nano": (settings.price_in_gpt5_nano, settings.price_out_gpt5_nano),
        "gpt-4o": (settings.price_in_gpt4o, settings.price_out_gpt4o),
    }
    key = "gpt-4o"
    for k in pricing:
        if model.startswith(k):
            key = k
            break
    pin, pout = pricing[key]
    return (in_tok * pin + out_tok * pout) / 1_000_000.0

# Helper to extract textual content robustly

def _extract_text(choices) -> str:
    def parts_to_str(parts):
        txts = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if t:
                    txts.append(str(t))
            else:
                txts.append(str(p))
        return "\n".join(txts).strip()

    if not choices:
        return ""
    primary = choices[0]
    msg = getattr(primary, "message", None)
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        text = parts_to_str(content)
    else:
        text = (content or "").strip()
    if not text:
        # check refusal or other attributes
        refusal = getattr(msg, "refusal", None)
        if refusal:
            text = str(refusal).strip()
    if not text and len(choices) > 1:
        for ch in choices[1:]:
            m2 = getattr(ch, "message", None)
            if not m2:
                continue
            c2 = getattr(m2, "content", None)
            if isinstance(c2, list):
                cand = parts_to_str(c2)
            else:
                cand = (c2 or "").strip()
            if cand:
                text = cand
                break
    if not text:
        log.warning("Respuesta del modelo sin contenido textual interpretable.")
    return text

@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(5))
def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    json_object: bool = False,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> tuple[str, LLMUsage]:
    """Directo a Chat Completions con robust text extraction y fallback de modelos."""
    messages = enforce_token_budget(messages)

    kwargs: Dict[str, Any] = dict(model=model, messages=messages)
    if temperature is not None and temperature != 1:
        kwargs["temperature"] = temperature
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    total_in = 0
    total_out = 0
    used_models: List[str] = []

    def _invoke(k: Dict[str, Any]):
        return client.chat.completions.create(**k)

    def _one_attempt(k: Dict[str, Any]) -> tuple[str, Any]:
        try:
            cc_local = _invoke(k)
        except Exception as e:
            se = str(e).lower()
            if "max_completion_tokens" in k and "max_tokens" in se and "unsupported" in se:
                k.pop("max_completion_tokens", None)
                if max_tokens is not None:
                    k["max_tokens"] = max_tokens
                cc_local = _invoke(k)
            elif "temperature" in k and "temperature" in se and "unsupported" in se:
                log.info("Retry sin temperature (no soportado por el modelo).")
                k.pop("temperature", None)
                cc_local = _invoke(k)
            else:
                raise
        text_local = _extract_text(cc_local.choices)
        usage_raw_local = cc_local.usage
        return text_local, usage_raw_local

    # Primary attempt
    text, usage_raw = _one_attempt(kwargs)
    used_models.append(model)
    total_in += getattr(usage_raw, "prompt_tokens", 0)
    total_out += getattr(usage_raw, "completion_tokens", 0)

    if json_object and not text.strip() and kwargs.get("response_format"):
        log.warning("Contenido vacío con response_format; reintentando sin JSON mode.")
        kwargs_no = dict(kwargs)
        kwargs_no.pop("response_format", None)
        t2, u2 = _one_attempt(kwargs_no)
        total_in += getattr(u2, "prompt_tokens", 0)
        total_out += getattr(u2, "completion_tokens", 0)
        if t2.strip():
            text = t2

    # Model fallback if still empty
    if not text.strip():
        alt_models = [m for m in ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13"] if m not in used_models]
        for alt in alt_models:
            log.warning(f"Texto vacío con modelo {model}; intentando fallback {alt}.")
            alt_kwargs = dict(kwargs)
            alt_kwargs["model"] = alt
            alt_kwargs.pop("response_format", None)  # evitar modo JSON en fallback
            t_alt, u_alt = _one_attempt(alt_kwargs)
            total_in += getattr(u_alt, "prompt_tokens", 0)
            total_out += getattr(u_alt, "completion_tokens", 0)
            used_models.append(alt)
            if t_alt.strip():
                text = t_alt
                break

    usage = LLMUsage(
        model=used_models[-1],
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=_estimate_cost(used_models[-1], total_in, total_out),
    )
    return text, usage

@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(5))
def call_llm_mm(
    model: str,
    parts: List[Dict[str, Any]],
    json_object: bool = True,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 800,
) -> tuple[str, LLMUsage]:
    """Multimodal con extraction y fallback de modelos."""
    content_parts: List[Dict[str, Any]] = []
    for p in parts:
        if p.get("type") == "input_text":
            content_parts.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "input_image":
            content_parts.append({"type": "image_url", "image_url": {"url": p.get("image_url", "")}})
    messages = [{"role": "user", "content": content_parts}]  # type: ignore

    kwargs: Dict[str, Any] = dict(model=model, messages=messages)
    if temperature is not None and temperature != 1:
        kwargs["temperature"] = temperature
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    total_in = 0
    total_out = 0
    used_models: List[str] = []

    def _invoke(k: Dict[str, Any]):
        return client.chat.completions.create(**k)

    def _one_attempt(k: Dict[str, Any]) -> tuple[str, Any]:
        try:
            cc_local = _invoke(k)
        except Exception as e:
            se = str(e).lower()
            if "max_completion_tokens" in k and "max_tokens" in se and "unsupported" in se:
                k.pop("max_completion_tokens", None)
                if max_tokens is not None:
                    k["max_tokens"] = max_tokens
                cc_local = _invoke(k)
            elif "temperature" in k and "temperature" in se and "unsupported" in se:
                log.info("Retry multimodal sin temperature (no soportado).")
                k.pop("temperature", None)
                cc_local = _invoke(k)
            else:
                raise RuntimeError(f"Multimodal call failed: {e}")
        txt_local = _extract_text(cc_local.choices)
        usage_raw_local = cc_local.usage
        return txt_local, usage_raw_local

    text, usage_raw = _one_attempt(kwargs)
    used_models.append(model)
    total_in += getattr(usage_raw, "prompt_tokens", 0)
    total_out += getattr(usage_raw, "completion_tokens", 0)

    if json_object and not text.strip() and kwargs.get("response_format"):
        log.warning("Contenido multimodal vacío con response_format; retry sin JSON mode.")
        kwargs_no = dict(kwargs)
        kwargs_no.pop("response_format", None)
        t2, u2 = _one_attempt(kwargs_no)
        total_in += getattr(u2, "prompt_tokens", 0)
        total_out += getattr(u2, "completion_tokens", 0)
        if t2.strip():
            text = t2

    if not text.strip():
        alt_models = [m for m in ["gpt-4o-mini", "gpt-4o"] if m not in used_models]
        for alt in alt_models:
            log.warning(f"Texto multimodal vacío con {model}; fallback {alt}.")
            alt_kwargs = dict(kwargs)
            alt_kwargs["model"] = alt
            alt_kwargs.pop("response_format", None)
            t_alt, u_alt = _one_attempt(alt_kwargs)
            total_in += getattr(u_alt, "prompt_tokens", 0)
            total_out += getattr(u_alt, "completion_tokens", 0)
            used_models.append(alt)
            if t_alt.strip():
                text = t_alt
                break

    usage = LLMUsage(
        model=used_models[-1],
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=_estimate_cost(used_models[-1], total_in, total_out),
    )
    return text, usage

from __future__ import annotations
from typing import Dict, Any, List
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm_mm
from app.guardrails.schemas import VisionStruct
from app.config import settings
from app.models.openai_llm import call_llm

SYS = (
    "You analyze an image and return a strict JSON with fields: contains_chart, contains_text, any_numbers, "
    "chart_type (optional), main_info (optional). Do NOT add extra fields."
)

class VisionAgent(BaseAgent):
    name = "vision"

    async def act(self):
        image_url = await self.bb.get("image_url")  # could be data URL or remote URL
        user_query = await self.bb.get("input") or ""
        if not image_url:
            # No image provided; set a neutral struct
            await self.bb.set("vision_struct", VisionStruct(
                contains_chart=False, contains_text=False, any_numbers=False
            ).model_dump())
            return

        parts = [
            {"type": "input_text", "text": f"Task: extract high-level info from the image for the query: {user_query}"},
            {"type": "input_image", "image_url": image_url},
        ]

        # try gpt-5 vision → fallback gpt-4o
        for model in (settings.model_vision, settings.model_vision_fallback):
            try:
                text, usage = call_llm_mm(model, parts, json_object=True, temperature=0.1, max_tokens=500)
                self.log.info(f"Vision via {model}")
                await self._record_usage(usage)
                out = VisionStruct.model_validate_json(text)
                await self.bb.set("vision_struct", out.model_dump())
                return
            except Exception as e:
                self.log.warning(f"Vision model {model} failed: {e}")
                continue

        # Hard fallback: textual description
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "No vision available; reply with contains_text=false unless certain."},
        ]
        text, usage = call_llm(settings.model_local_fallback, msgs, json_object=True)
        await self._record_usage(usage)
        try:
            out = VisionStruct.model_validate_json(text)
        except ValidationError:
            out = VisionStruct(contains_chart=False, contains_text=False, any_numbers=False)
        await self.bb.set("vision_struct", out.model_dump())

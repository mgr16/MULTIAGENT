from __future__ import annotations
from typing import Dict, List
from pydantic import ValidationError
from app.agents.base import BaseAgent
from app.models.openai_llm import call_llm
from app.guardrails.schemas import PlanOutput, PlanStep

SYS = (
    "You design multi-agent plans as a DAG. Output strictly JSON matching PlanOutput schema. "
    "Prefer 3-6 steps. Use parallel_group to run steps simultaneously when possible. "
    "Stop when 'final_answer' is produced."
)

TEMPLATE = """User query:
{query}

Domain: {domain}

Agents available: vision, rag, web_search, data, critic, summary, hypothesis, memory, local_text

Design a plan with steps and parallel groups. Include prerequisites in 'requires'."""

class PlannerAgent(BaseAgent):
    name = "planner"

    async def act(self):
        ro = await self.bb.get("router_output") or {"domain":"general"}
        user_query = await self.bb.get("input") or ""
        content = TEMPLATE.format(query=user_query, domain=ro.get("domain","general"))
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": content},
        ]
        model = self.choose_model(importance="high", default="gpt-5")
        text, usage = call_llm(model, msgs, json_object=True, temperature=0.2, max_tokens=900)
        await self._record_usage(usage)
        try:
            plan = PlanOutput.model_validate_json(text)
        except ValidationError:
            # Fallback determinista
            steps = [
                PlanStep(name="gather", agents=["rag","web_search"], requires=[]).model_dump(),
                PlanStep(name="analyze", agents=["data"], requires=["gather"]).model_dump(),
                PlanStep(name="draft", agents=["summary"], requires=["gather","analyze"]).model_dump(),
                PlanStep(name="critique", agents=["critic"], requires=["draft"]).model_dump(),
                PlanStep(name="finalize", agents=["summary"], requires=["critique"]).model_dump(),
            ]
            plan = PlanOutput(steps=[PlanStep(**s) for s in steps], stop_condition="final_answer")
        await self.bb.set("plan", plan.model_dump())

        # Derivar "plan_layers" (lista de capas paralelas por parallel_group o por orden)
        layers: List[List[str]] = []
        # Construcción simple: si parallel_group existe, agrupa por ese nombre en orden de aparición
        groups: Dict[str, List[str]] = {}
        order: List[str] = []
        for s in plan.steps:
            pg = s.parallel_group or s.name
            if pg not in groups:
                groups[pg] = []
                order.append(pg)
            groups[pg].extend(s.agents)
        for g in order:
            layers.append(groups[g])
        await self.bb.set("plan_layers", layers)

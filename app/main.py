import asyncio
from typing import List, Callable, Dict
from app.blackboard import Blackboard
from app.scheduler import Scheduler
from app.logging_setup import get_logger
from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent
from app.agents.vision import VisionAgent
from app.agents.rag import RAGAgent
from app.agents.web_search import WebSearchAgent
from app.agents.data_analysis import DataAnalysisAgent
from app.agents.critic import CriticAgent
from app.agents.summary import SummaryAgent
from app.agents.hypothesis import HypothesisAgent
from app.agents.memory import MemoryAgent
from app.agents.local_text import LocalTextAgent

log = get_logger("main")

def build_agent_registry(bb: Blackboard) -> Dict[str, Callable[[], None]]:
    router = RouterAgent(bb)
    planner = PlannerAgent(bb)
    vision = VisionAgent(bb)
    rag = RAGAgent(bb)
    web = WebSearchAgent(bb)
    data = DataAnalysisAgent(bb)
    critic = CriticAgent(bb)
    summary = SummaryAgent(bb)
    hypothesis = HypothesisAgent(bb)
    memory = MemoryAgent(bb)
    local_text = LocalTextAgent(bb)

    # map nombre → callable async
    return {
        "router": router.act,
        "planner": planner.act,
        "vision": vision.act,
        "rag": rag.act,
        "web_search": web.act,
        "data": data.act,
        "critic": critic.act,
        "summary": summary.act,
        "hypothesis": hypothesis.act,
        "memory": memory.act,
        "local_text": local_text.act,
    }

def layers_to_callables(layers: List[List[str]], registry: Dict[str, Callable], bb: Blackboard):
    plan: List[List[Callable]] = []
    seen_summary = False
    for layer in layers:
        fns: List[Callable] = []
        # If critic in this layer and we already had a summary layer before, inject promotion
        if "critic" in layer and seen_summary:
            async def promote_draft():  # closure over bb
                if not await bb.get("final_answer"):
                    draft = await bb.get("draft_answer") or {}
                    if draft:
                        await bb.set("final_answer", draft.get("final_answer", ""))
            fns.append(promote_draft)
        for name in layer:
            if name not in registry:
                log.warning(f"Unknown agent '{name}' in plan. Skipping.")
                continue
            fns.append(registry[name])
            if name == "summary":
                seen_summary = True
        plan.append(fns)
    return plan

async def run_query(query: str, image_url: str | None = None):
    bb = Blackboard()
    await bb.set("input", query)
    if image_url:
        await bb.set("image_url", image_url)

    registry = build_agent_registry(bb)
    sched = Scheduler(bb)

    # Bootstrap: router → planner → rest (a partir de su plan)
    await registry["router"]()
    await registry["planner"]()

    layers = await bb.get("plan_layers") or [["rag","web_search"], ["data"], ["summary"], ["critic"], ["summary"], ["memory"]]
    plan = layers_to_callables(layers, registry, bb)

    await sched.run(plan)

    fa = await bb.get("final_answer")
    if not fa:
        draft = await bb.get("draft_answer")
        if draft:
            await bb.set("final_answer", draft.get("final_answer", ""))

    verdict = await bb.get("critic_verdict")
    if verdict and verdict.get("conflicts", 0) > 0:
        log.info("Conflicts detected; running final summary refinement.")
        await registry["summary"]()

    final_answer = await bb.get("final_answer") or (await bb.get("draft_answer") or {}).get("final_answer")
    usage_log = await bb.get("usage_log") or []
    return final_answer, usage_log

async def demo():
    ans, usage = await run_query("Resume los beneficios y riesgos de usar modelos locales vs GPT-5 para análisis financiero, cita fuentes si puedes.")
    print("=== FINAL ANSWER ===")
    print(ans)
    print("\n=== USAGE ===")
    print("\n".join(usage))

if __name__ == "__main__":
    asyncio.run(demo())

import pytest
from app.blackboard import Blackboard
from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent

@pytest.mark.asyncio
async def test_planner_layers():
    bb = Blackboard()
    await bb.set("input", "Comparar modelos locales vs GPT-5.")
    await RouterAgent(bb).act()
    await PlannerAgent(bb).act()
    layers = await bb.get("plan_layers")
    assert isinstance(layers, list) and len(layers) >= 3

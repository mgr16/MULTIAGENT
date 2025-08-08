import pytest
from app.blackboard import Blackboard
from app.agents.router import RouterAgent

@pytest.mark.asyncio
async def test_router_basic():
    bb = Blackboard()
    await bb.set("input", "Explícame qué es una curva de rendimiento invertida.")
    ag = RouterAgent(bb)
    await ag.act()
    out = await bb.get("router_output")
    assert out and out["domain"] == "general" and "planner" in out["suggested_agents"]

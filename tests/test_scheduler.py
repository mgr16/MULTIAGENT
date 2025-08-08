import pytest
import asyncio
from app.blackboard import Blackboard
from app.scheduler import Scheduler

@pytest.mark.asyncio
async def test_scheduler_parallel_cancel():
    bb = Blackboard()
    sched = Scheduler(bb)

    async def slow():
        await asyncio.sleep(0.2)
        await bb.set("x", 1)

    async def finalizer():
        await asyncio.sleep(0.05)
        await bb.set("final_answer", "ok")

    plan = [[slow, finalizer]]
    await sched.run(plan)
    fa = await bb.get("final_answer")
    assert fa == "ok"

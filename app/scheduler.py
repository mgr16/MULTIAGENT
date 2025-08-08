from __future__ import annotations
import asyncio
from typing import Dict, List, Callable, Any
from app.blackboard import Blackboard
from app.logging_setup import get_logger

log = get_logger("scheduler")

class Scheduler:
    """
    Ejecuta pasos en paralelo y cancela cuando encuentra 'final_answer'.
    """
    def __init__(self, bb: Blackboard):
        self.bb = bb
        self.tasks: List[asyncio.Task] = []
        self.cancel_event = asyncio.Event()

    async def run_parallel(self, steps: List[Callable[[], Any]]):
        async with asyncio.TaskGroup() as tg:  # Python 3.11+
            for step in steps:
                tg.create_task(step())

    async def run(self, plan: List[List[Callable[[], Any]]]):
        for layer in plan:
            if self.cancel_event.is_set():
                log.info("Cancelado por finalización anticipada.")
                break
            await self.run_parallel(layer)
            if await self.bb.exists("final_answer"):
                self.cancel_event.set()
                log.info("Final answer presente; fin del grafo.")

    def cancel(self):
        self.cancel_event.set()

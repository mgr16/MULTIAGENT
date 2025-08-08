import pytest
from app.main import run_query

@pytest.mark.asyncio
async def test_end_to_end_smoke():
    ans, usage = await run_query("Diferencias entre supervisado, no supervisado, semi-supervisado y RL.", image_url=None)
    assert isinstance(ans, str) and len(ans) > 0
    assert isinstance(usage, list)

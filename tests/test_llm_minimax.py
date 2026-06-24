import os, pytest
from pixelle_video.services.llm_service import LLMService

@pytest.mark.asyncio
async def test_minimax_m3_live():
    if not os.getenv("MINIMAX_API_KEY"):
        pytest.skip("no MINIMAX_API_KEY")
    svc = LLMService({})
    out = await svc(
        prompt="用一句廣東話講解咩係光合作用。",
        api_key=os.getenv("MINIMAX_API_KEY"),
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M3",
        max_tokens=200,
    )
    assert isinstance(out, str) and len(out) > 5
    print("\nMiniMax-M3 回應:", out[:200])

# tests/test_api_media_openrouter_video.py
from unittest.mock import patch
import pytest
from pixelle_video.services.api_media import APIProviderMediaService


@pytest.mark.asyncio
async def test_dispatch_openrouter_video(tmp_path):
    svc = APIProviderMediaService({})
    out = tmp_path / "v.mp4"
    with patch("pixelle_video.services.api_media.OpenRouterVideoClient") as Cli:
        def _gen(**kw): out.write_bytes(b"MP4"); return "http://x/v.mp4"
        Cli.return_value.generate_video.side_effect = _gen
        res = await svc(prompt="cat", workflow="api/openrouter/bytedance/seedance-2.0",
                        media_type="video", output_path=str(out), duration=5.0)
        assert res.is_video
        Cli.return_value.generate_video.assert_called_once()

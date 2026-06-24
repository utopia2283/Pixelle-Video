from unittest.mock import patch
import pytest
from pixelle_video.services.api_media import APIProviderMediaService


def test_resolve_openrouter_image_key():
    svc = APIProviderMediaService({})
    info = svc.resolve_workflow("api/openrouter/bytedance-seed/seedream-4.5")
    assert info["provider"] == "openrouter"
    assert info["model"] == "bytedance-seed/seedream-4.5"
    assert info["media_type"] == "image"


@pytest.mark.asyncio
async def test_dispatch_openrouter_image(tmp_path):
    svc = APIProviderMediaService({})
    with patch("pixelle_video.services.api_media.OpenRouterImageClient") as Cli:
        Cli.return_value.generate_image.return_value = [str(tmp_path / "x.png")]
        (tmp_path / "x.png").write_bytes(b"\x89PNG")
        res = await svc(prompt="hi", workflow="api/openrouter/bytedance-seed/seedream-4.5",
                        media_type="image", output_path=str(tmp_path / "x.png"))
        assert res.is_image
        Cli.return_value.generate_image.assert_called_once()

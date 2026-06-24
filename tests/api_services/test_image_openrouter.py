import base64, json
from unittest.mock import patch, MagicMock
from pixelle_video.services.api_services.image_openrouter import OpenRouterImageClient


def _png_b64():
    # 1x1 px PNG
    return base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415408d763f8cfc0f01f0005000100ffa9b4c40000000049454e44ae426082"
    )).decode()


@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_generate_image_t2i(mock_post, tmp_path):
    mock_post.return_value = MagicMock(status_code=200,
        json=lambda: {"data": [{"b64_json": _png_b64()}]})
    mock_post.return_value.raise_for_status = lambda: None
    c = OpenRouterImageClient(api_key="sk-test")
    paths = c.generate_image("a red panda", save_dir=str(tmp_path),
                             model="bytedance-seed/seedream-4.5", resolution="2K", video_ratio="16:9")
    assert len(paths) == 1
    assert paths[0].endswith(".png") or paths[0].endswith(".jpg")
    # 驗 request shape
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "bytedance-seed/seedream-4.5"
    assert body["prompt"] == "a red panda"
    assert body["resolution"] == "2K"
    assert body["aspect_ratio"] == "16:9"
    assert "input_references" not in body


@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_generate_image_i2i(mock_post, tmp_path):
    ref = tmp_path / "ref.png"; ref.write_bytes(base64.b64decode(_png_b64()))
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"data": [{"b64_json": _png_b64()}]})
    mock_post.return_value.raise_for_status = lambda: None
    c = OpenRouterImageClient(api_key="sk-test")
    c.generate_image("stylize", image_paths=[str(ref)], save_dir=str(tmp_path))
    refs = mock_post.call_args.kwargs["json"]["input_references"]
    assert refs[0]["image_url"]["url"].startswith("data:image/")

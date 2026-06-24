"""Tests for OpenRouterImageClient using Grok chat/completions endpoint."""
import base64
import os
from unittest.mock import patch, MagicMock

import pytest

from pixelle_video.services.api_services.image_openrouter import OpenRouterImageClient


# Minimal valid 1x1 PNG bytes
_PNG_HEX = (
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0f01f0005000100ffa9b4c40000000049454e44ae426082"
)


def _png_bytes() -> bytes:
    return bytes.fromhex(_PNG_HEX)


def _png_b64() -> str:
    return base64.b64encode(_png_bytes()).decode()


def _png_data_uri() -> str:
    return f"data:image/png;base64,{_png_b64()}"


def _mock_chat_response(image_url: str):
    """Build a mock response matching Grok chat/completions image response shape."""
    return {
        "choices": [
            {
                "message": {
                    "images": [
                        {"image_url": {"url": image_url}}
                    ]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# (a) text-to-image: modalities == ["image"], no "text" in modalities,
#     first content element is a text message
# ---------------------------------------------------------------------------
@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_t2i_request_body_modalities_and_content(mock_post, tmp_path):
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = _mock_chat_response(_png_data_uri())
    mock_post.return_value = mock_resp

    client = OpenRouterImageClient(api_key="sk-test")
    client.generate_image("a red panda", save_dir=str(tmp_path))

    call_kwargs = mock_post.call_args.kwargs
    body = call_kwargs["json"]

    # endpoint must be chat/completions, not /images
    url = mock_post.call_args.args[0] if mock_post.call_args.args else call_kwargs.get("url", "")
    assert "/chat/completions" in url

    # modalities must be exactly ["image"] — "text" must NOT be present
    assert body["modalities"] == ["image"]
    assert "text" not in body["modalities"]

    # messages structure
    assert body["messages"][0]["role"] == "user"
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "a red panda"


# ---------------------------------------------------------------------------
# (b) image is decoded, written to disk, and file content is correct
# ---------------------------------------------------------------------------
@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_t2i_image_saved_correctly(mock_post, tmp_path):
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = _mock_chat_response(_png_data_uri())
    mock_post.return_value = mock_resp

    client = OpenRouterImageClient(api_key="sk-test")
    paths = client.generate_image("a red panda", save_dir=str(tmp_path))

    assert len(paths) == 1
    out_path = paths[0]
    assert os.path.exists(out_path)
    assert open(out_path, "rb").read() == _png_bytes()


# ---------------------------------------------------------------------------
# (c) image-to-image: content array second element is image_url with data URI
# ---------------------------------------------------------------------------
@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_i2i_content_includes_image_url(mock_post, tmp_path):
    # Write a real PNG as the reference image
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(_png_bytes())

    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = _mock_chat_response(_png_data_uri())
    mock_post.return_value = mock_resp

    client = OpenRouterImageClient(api_key="sk-test")
    client.generate_image("stylize it", image_paths=[str(ref_path)], save_dir=str(tmp_path))

    body = mock_post.call_args.kwargs["json"]
    content = body["messages"][0]["content"]

    # Second element must be image_url type
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    image_url_val = content[1]["image_url"]["url"]
    assert image_url_val.startswith("data:image/")


# ---------------------------------------------------------------------------
# default model is grok, not seedream
# ---------------------------------------------------------------------------
@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_default_model_is_grok(mock_post, tmp_path):
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = _mock_chat_response(_png_data_uri())
    mock_post.return_value = mock_resp

    client = OpenRouterImageClient(api_key="sk-test")
    client.generate_image("test prompt", save_dir=str(tmp_path))

    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "x-ai/grok-imagine-image-quality"


# ---------------------------------------------------------------------------
# video_ratio and resolution are accepted in the signature but not sent in body
# ---------------------------------------------------------------------------
@patch("pixelle_video.services.api_services.image_openrouter.requests.post")
def test_video_ratio_and_resolution_not_in_body(mock_post, tmp_path):
    mock_resp = MagicMock(status_code=200)
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = _mock_chat_response(_png_data_uri())
    mock_post.return_value = mock_resp

    client = OpenRouterImageClient(api_key="sk-test")
    client.generate_image(
        "test prompt",
        save_dir=str(tmp_path),
        video_ratio="16:9",
        resolution="2K",
    )

    body = mock_post.call_args.kwargs["json"]
    assert "resolution" not in body
    assert "aspect_ratio" not in body
    assert "video_ratio" not in body

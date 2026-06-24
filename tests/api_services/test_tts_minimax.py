# tests/api_services/test_tts_minimax.py
"""Tests for MiniMax TTS client and TTSService minimax mode."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from pixelle_video.services.api_services.tts_minimax import MiniMaxTTSClient


@patch("pixelle_video.services.api_services.tts_minimax.requests.post")
def test_synthesize(mock_post, tmp_path):
    """Test synthesize writes hex-decoded audio bytes to output path."""
    audio_hex = b"ID3test".hex()
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {"audio": audio_hex, "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    mock_post.return_value.raise_for_status = lambda: None

    c = MiniMaxTTSClient(api_key="mm-test")
    out = tmp_path / "a.mp3"
    p = c.synthesize("你好", str(out), voice="Cantonese_GentleLady", speed=1.0)

    assert out.read_bytes() == b"ID3test"
    assert p == str(out)
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "speech-2.8-turbo"
    assert body["text"] == "你好"
    assert body["voice_setting"]["voice_id"] == "Cantonese_GentleLady"
    assert body["output_format"] == "hex"
    assert body["language_boost"] == "Chinese,Yue"
    assert body["audio_setting"]["format"] == "mp3"


@patch("pixelle_video.services.api_services.tts_minimax.requests.post")
def test_synthesize_api_error_raises(mock_post, tmp_path):
    """Test that non-zero base_resp.status_code raises RuntimeError."""
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {},
            "base_resp": {"status_code": 1001, "status_msg": "invalid key"},
        },
    )
    mock_post.return_value.raise_for_status = lambda: None

    c = MiniMaxTTSClient(api_key="mm-test")
    with pytest.raises(RuntimeError, match="MiniMax TTS error"):
        c.synthesize("hello", str(tmp_path / "out.mp3"), voice="Cantonese_GentleLady")


@patch("pixelle_video.services.api_services.tts_minimax.requests.post")
def test_synthesize_no_api_key_raises(mock_post, tmp_path, monkeypatch):
    """Test that missing API key raises RuntimeError before any HTTP call."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    c = MiniMaxTTSClient(api_key=None)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY not set"):
        c.synthesize("hello", str(tmp_path / "out.mp3"), voice="Cantonese_GentleLady")
    mock_post.assert_not_called()


@patch("pixelle_video.services.api_services.tts_minimax.requests.post")
def test_synthesize_group_id_appended_to_url(mock_post, tmp_path):
    """Test that group_id is appended as query param when provided."""
    audio_hex = b"audio".hex()
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {"audio": audio_hex},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    mock_post.return_value.raise_for_status = lambda: None

    c = MiniMaxTTSClient(api_key="mm-test", group_id="grp-123")
    c.synthesize("hi", str(tmp_path / "b.mp3"), voice="Cantonese_GentleLady")

    called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    # When using requests.post(url, ...) url is positional arg
    assert "GroupId=grp-123" in called_url


@patch("pixelle_video.services.api_services.tts_minimax.requests.post")
def test_synthesize_no_group_id_url_clean(mock_post, tmp_path):
    """Test that URL has no GroupId param when group_id is not set."""
    audio_hex = b"audio".hex()
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {"audio": audio_hex},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    mock_post.return_value.raise_for_status = lambda: None

    c = MiniMaxTTSClient(api_key="mm-test", group_id=None)
    c.synthesize("hi", str(tmp_path / "c.mp3"), voice="Cantonese_GentleLady")

    called_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    assert "GroupId" not in called_url


# TTSService integration: minimax mode routes to MiniMaxTTSClient
@pytest.mark.asyncio
@patch("pixelle_video.services.tts_service.MiniMaxTTSClient")
async def test_tts_service_minimax_mode(MockClient, tmp_path):
    """Test TTSService routes to MiniMaxTTSClient when mode='minimax'."""
    from pixelle_video.services.tts_service import TTSService

    mock_instance = MagicMock()
    mock_instance.synthesize.return_value = str(tmp_path / "out.mp3")
    MockClient.return_value = mock_instance

    # ComfyBaseService.__init__ sets self.config = config["comfyui"]["tts"]
    # so inference_mode must live under that key
    config = {
        "comfyui": {
            "tts": {
                "inference_mode": "minimax",
                "minimax": {
                    "voice": "Cantonese_GentleLady",
                    "speed": 1.0,
                    "model": "speech-2.8-turbo",
                },
            }
        },
    }

    # TTSService needs a core with _get_or_create_comfykit for comfyui mode,
    # but minimax mode should NOT call that — pass None
    svc = TTSService(config=config, core=None)
    result = await svc("你好世界", output_path=str(tmp_path / "out.mp3"))

    mock_instance.synthesize.assert_called_once()
    call_kwargs = mock_instance.synthesize.call_args
    assert call_kwargs.kwargs["text"] == "你好世界" or call_kwargs.args[0] == "你好世界"
    assert result == str(tmp_path / "out.mp3")

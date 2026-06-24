# tests/api_services/test_video_openrouter.py
import pytest
from unittest.mock import patch, MagicMock, call
from pixelle_video.services.api_services.video_openrouter import OpenRouterVideoClient


def _make_dl_mock(data: bytes = b"MP4DATA"):
    """Return a streaming-compatible mock for the download response."""
    dl = MagicMock(status_code=200)
    dl.raise_for_status = lambda: None
    # iter_content must return an iterable of chunks
    dl.iter_content = lambda chunk_size=8192: iter([data])
    return dl


@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_generate_video_flow(mock_post, mock_get, tmp_path):
    """Happy-path: text-to-video, no image_path."""
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "job_1"})
    mock_post.return_value.raise_for_status = lambda: None

    poll = MagicMock(status_code=200, json=lambda: {"status": "completed", "output": {"url": "http://x/v.mp4"}})
    poll.raise_for_status = lambda: None

    # New download flow: first call is WITHOUT auth (no-auth attempt) and succeeds (200)
    dl = _make_dl_mock(b"MP4DATA")
    mock_get.side_effect = [poll, dl]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "v.mp4"
    url = c.generate_video("a cat", image_path=None, save_path=str(out),
                           model="bytedance/seedance-2.0", duration=5)
    assert out.read_bytes() == b"MP4DATA"
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "bytedance/seedance-2.0"
    assert body["prompt"] == "a cat"
    assert body["duration"] == 5

    # Download should NOT send Authorization header on the first (no-auth) attempt
    download_call_kwargs = mock_get.call_args_list[1].kwargs
    download_headers = download_call_kwargs.get("headers") or {}
    assert "Authorization" not in download_headers, (
        "First download attempt must NOT include Authorization header (CDN pre-signed URL may reject it)"
    )


@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_download_retries_with_auth_on_403(mock_post, mock_get, tmp_path):
    """I-1: if CDN returns 403, retry download WITH Authorization header."""
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "job_2"})
    mock_post.return_value.raise_for_status = lambda: None

    poll = MagicMock(status_code=200, json=lambda: {"status": "completed", "output": {"url": "http://cdn/v.mp4"}})
    poll.raise_for_status = lambda: None

    # First download attempt: 403 (CDN rejects no-auth).
    # raise_for_status is NOT called on this object — the implementation checks
    # status_code directly and retries before calling raise_for_status.
    dl_forbidden = MagicMock(status_code=403)
    dl_forbidden.raise_for_status = lambda: None  # won't be called

    # Second download attempt: 200 with auth header
    dl_ok = _make_dl_mock(b"VIDEODATA")

    mock_get.side_effect = [poll, dl_forbidden, dl_ok]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "retry.mp4"
    url = c.generate_video("a dog", image_path=None, save_path=str(out))
    assert out.read_bytes() == b"VIDEODATA"

    # Third get call (index 2) must include Authorization header
    retry_call_kwargs = mock_get.call_args_list[2].kwargs
    retry_headers = retry_call_kwargs.get("headers") or {}
    assert "Authorization" in retry_headers, (
        "Retry download after 403 must include Authorization header"
    )


@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_frame_images_in_submit_body(mock_post, mock_get, tmp_path):
    """m-2(a): when image_path is provided, submit body must contain frame_images
    with the first element having frame_type == 'first_frame'."""
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "job_3"})
    mock_post.return_value.raise_for_status = lambda: None

    poll = MagicMock(status_code=200, json=lambda: {"status": "completed", "output": {"url": "http://x/v.mp4"}})
    poll.raise_for_status = lambda: None
    dl = _make_dl_mock()
    mock_get.side_effect = [poll, dl]

    # Create a tiny fake PNG file
    fake_image = tmp_path / "frame.png"
    # Minimal valid PNG bytes (1×1 white pixel)
    fake_image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "out.mp4"
    c.generate_video("sunrise", image_path=str(fake_image), save_path=str(out))

    body = mock_post.call_args.kwargs["json"]
    assert "frame_images" in body, "Submit body must contain 'frame_images' when image_path is set"
    frames = body["frame_images"]
    assert len(frames) >= 1, "frame_images must have at least one entry"
    assert frames[0]["frame_type"] == "first_frame", (
        f"First frame must have frame_type='first_frame', got {frames[0].get('frame_type')!r}"
    )


@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_poll_failed_raises_runtime_error(mock_post, mock_get, tmp_path):
    """m-2(b): poll response with status='failed' must raise RuntimeError."""
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "job_4"})
    mock_post.return_value.raise_for_status = lambda: None

    poll_failed = MagicMock(
        status_code=200,
        json=lambda: {"status": "failed", "error": "something went wrong"},
    )
    poll_failed.raise_for_status = lambda: None
    mock_get.side_effect = [poll_failed]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "fail.mp4"
    with pytest.raises(RuntimeError, match="failed"):
        c.generate_video("bad prompt", image_path=None, save_path=str(out))

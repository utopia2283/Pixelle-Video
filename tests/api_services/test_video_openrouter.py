# tests/api_services/test_video_openrouter.py
"""TDD tests for OpenRouterVideoClient (Seedance 1.5 Pro + tmpfiles public-URL frames).

All tests are fully mocked — no real API calls are made.

Mock order for requests.post side_effect:
  call 0 → tmpfiles upload (returns {"data":{"url":"https://tmpfiles.org/12345/x.png"}})
  call 1 → OpenRouter submit (returns {"polling_url":"https://openrouter.ai/api/v1/videos/job_1"})

Mock order for requests.get side_effect:
  call 0 → poll (returns {"status":"completed","unsigned_urls":["https://cdn/v.mp4"]})
  call 1 → download (returns streaming 200 response)
"""

import pytest
from unittest.mock import patch, MagicMock
from pixelle_video.services.api_services.video_openrouter import OpenRouterVideoClient


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_dl_mock(data: bytes = b"MP4DATA"):
    """Return a streaming-compatible mock for the download response."""
    dl = MagicMock(status_code=200)
    dl.raise_for_status = lambda: None
    dl.iter_content = lambda chunk_size=8192: iter([data])
    return dl


def _tmpfiles_post_mock():
    """Mock for the tmpfiles.org upload POST call."""
    m = MagicMock(status_code=200)
    m.raise_for_status = lambda: None
    m.json = lambda: {"data": {"url": "https://tmpfiles.org/12345/x.png"}}
    return m


def _submit_post_mock(polling_url="https://openrouter.ai/api/v1/videos/job_1"):
    """Mock for the OpenRouter submit POST call."""
    m = MagicMock(status_code=200)
    m.raise_for_status = lambda: None
    m.json = lambda: {"polling_url": polling_url}
    return m


def _poll_get_mock(status="completed", unsigned_urls=None):
    """Mock for an OpenRouter poll GET call."""
    if unsigned_urls is None:
        unsigned_urls = ["https://cdn/v.mp4"]
    m = MagicMock(status_code=200)
    m.raise_for_status = lambda: None
    m.json = lambda: {"status": status, "unsigned_urls": unsigned_urls}
    return m


# ── test: default model is seedance-1-5-pro ──────────────────────────────────

def test_default_model_is_seedance_1_5_pro(tmp_path):
    """The default model parameter must be bytedance/seedance-1-5-pro."""
    import inspect
    from pixelle_video.services.api_services.video_openrouter import OpenRouterVideoClient
    sig = inspect.signature(OpenRouterVideoClient.generate_video)
    assert sig.parameters["model"].default == "bytedance/seedance-1-5-pro", (
        f"Expected default model 'bytedance/seedance-1-5-pro', got {sig.parameters['model'].default!r}"
    )


# ── test: text-to-video happy path (no image) ─────────────────────────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_text_to_video_no_image(mock_post, mock_get, tmp_path):
    """Happy path: text-to-video, no image_path — no tmpfiles upload, uses polling_url."""
    # Only one POST: the OpenRouter submit
    mock_post.side_effect = [_submit_post_mock()]
    poll = _poll_get_mock()
    dl = _make_dl_mock(b"MP4DATA")
    mock_get.side_effect = [poll, dl]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "v.mp4"
    url = c.generate_video("a cat", image_path=None, save_path=str(out),
                           model="bytedance/seedance-1-5-pro", duration=5)

    assert out.read_bytes() == b"MP4DATA"
    # POST was called exactly once (submit only — no tmpfiles upload for text-to-video)
    assert mock_post.call_count == 1

    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "bytedance/seedance-1-5-pro"
    assert body["prompt"] == "a cat"
    assert body["duration"] == 5


# ── test (a): frame_images use tmpfiles public URL, not data URI ──────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_frame_images_use_tmpfiles_public_url(mock_post, mock_get, tmp_path):
    """(a) When image_path provided, frame is uploaded to tmpfiles and frame_images[0].image_url.url
    is the converted direct-link URL (https://tmpfiles.org/dl/...), NOT a data: URI."""

    # POST call order: 0=tmpfiles upload, 1=OpenRouter submit
    mock_post.side_effect = [_tmpfiles_post_mock(), _submit_post_mock()]
    poll = _poll_get_mock()
    dl = _make_dl_mock()
    mock_get.side_effect = [poll, dl]

    # Fake PNG image
    fake_image = tmp_path / "frame.png"
    fake_image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "out.mp4"
    c.generate_video("sunrise", image_path=str(fake_image), save_path=str(out))

    # Two POST calls: tmpfiles then submit
    assert mock_post.call_count == 2, (
        f"Expected 2 POST calls (tmpfiles + submit), got {mock_post.call_count}"
    )

    submit_body = mock_post.call_args_list[1].kwargs["json"]
    assert "frame_images" in submit_body, "Submit body must contain 'frame_images'"
    frames = submit_body["frame_images"]
    assert len(frames) >= 1
    assert frames[0]["frame_type"] == "first_frame"

    frame_url = frames[0]["image_url"]["url"]

    # Must NOT be a data URI
    assert not frame_url.startswith("data:"), (
        f"frame_images[0].image_url.url must NOT be a data: URI, got: {frame_url!r}"
    )

    # Must be the converted tmpfiles direct-link URL
    assert frame_url == "https://tmpfiles.org/dl/12345/x.png", (
        f"Expected 'https://tmpfiles.org/dl/12345/x.png', got {frame_url!r}"
    )

    # Must be an http(s) public URL
    assert frame_url.startswith("https://"), (
        f"frame_images URL must be a public https URL, got: {frame_url!r}"
    )


# ── test: tmpfiles URL conversion inserts /dl after host ─────────────────────

def test_tmpfiles_url_conversion():
    """_tmpfiles_direct_url converts https://tmpfiles.org/12345/x.png
    to https://tmpfiles.org/dl/12345/x.png."""
    c = OpenRouterVideoClient(api_key="sk-test")
    result = c._tmpfiles_direct_url("https://tmpfiles.org/12345/x.png")
    assert result == "https://tmpfiles.org/dl/12345/x.png"


# ── test: last_image_path also uses tmpfiles upload ──────────────────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_last_frame_also_uploaded_via_tmpfiles(mock_post, mock_get, tmp_path):
    """Both first_frame and last_frame are uploaded via tmpfiles."""
    tmpfiles1 = MagicMock(status_code=200)
    tmpfiles1.raise_for_status = lambda: None
    tmpfiles1.json = lambda: {"data": {"url": "https://tmpfiles.org/001/first.png"}}

    tmpfiles2 = MagicMock(status_code=200)
    tmpfiles2.raise_for_status = lambda: None
    tmpfiles2.json = lambda: {"data": {"url": "https://tmpfiles.org/002/last.png"}}

    # POST order: tmpfiles for first, tmpfiles for last, then submit
    mock_post.side_effect = [tmpfiles1, tmpfiles2, _submit_post_mock()]
    poll = _poll_get_mock()
    dl = _make_dl_mock()
    mock_get.side_effect = [poll, dl]

    fake_image = tmp_path / "first.png"
    fake_image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    fake_last = tmp_path / "last.png"
    fake_last.write_bytes(fake_image.read_bytes())

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "out.mp4"
    c.generate_video("test", image_path=str(fake_image), last_image_path=str(fake_last),
                     save_path=str(out))

    assert mock_post.call_count == 3  # 2 tmpfiles + 1 submit

    submit_body = mock_post.call_args_list[2].kwargs["json"]
    frames = submit_body["frame_images"]
    assert len(frames) == 2
    assert frames[0]["frame_type"] == "first_frame"
    assert frames[1]["frame_type"] == "last_frame"
    for fr in frames:
        assert not fr["image_url"]["url"].startswith("data:")
        assert "tmpfiles.org/dl/" in fr["image_url"]["url"]


# ── test (b): poll uses polling_url from submit response ─────────────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_poll_uses_polling_url_from_submit_response(mock_post, mock_get, tmp_path):
    """(b) Polling GET must be sent to the polling_url returned in the submit response."""
    polling_url = "https://openrouter.ai/api/v1/videos/job_XYZ"
    mock_post.side_effect = [_submit_post_mock(polling_url=polling_url)]
    poll = _poll_get_mock()
    dl = _make_dl_mock()
    mock_get.side_effect = [poll, dl]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "poll_test.mp4"
    c.generate_video("test", image_path=None, save_path=str(out))

    # First GET call must be to polling_url
    first_get_url = mock_get.call_args_list[0].args[0]
    assert first_get_url == polling_url, (
        f"Poll must use polling_url={polling_url!r}, got {first_get_url!r}"
    )


# ── test (c): download uses unsigned_urls[0] with Authorization header ────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_download_uses_unsigned_urls_with_auth(mock_post, mock_get, tmp_path):
    """(c) Download must use unsigned_urls[0] and include Authorization header."""
    video_url = "https://cdn.example.com/signed/video.mp4"
    mock_post.side_effect = [_submit_post_mock()]
    poll = _poll_get_mock(unsigned_urls=[video_url])
    dl = _make_dl_mock(b"VIDEODATA")
    mock_get.side_effect = [poll, dl]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "dl_test.mp4"
    c.generate_video("test", image_path=None, save_path=str(out))

    assert out.read_bytes() == b"VIDEODATA"

    # Second GET call is the download
    dl_call = mock_get.call_args_list[1]
    dl_url = dl_call.args[0]
    dl_headers = dl_call.kwargs.get("headers") or {}

    assert dl_url == video_url, f"Download URL must be unsigned_urls[0]={video_url!r}, got {dl_url!r}"
    assert "Authorization" in dl_headers, (
        "Download must include Authorization header (Bearer token) for unsigned_urls"
    )
    assert dl_headers["Authorization"].startswith("Bearer "), (
        f"Authorization must be a Bearer token, got: {dl_headers['Authorization']!r}"
    )


# ── test (d): status "failed" raises RuntimeError ────────────────────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_poll_failed_raises_runtime_error(mock_post, mock_get, tmp_path):
    """(d) poll response with status='failed' must raise RuntimeError."""
    mock_post.side_effect = [_submit_post_mock()]
    poll_failed = MagicMock(status_code=200)
    poll_failed.raise_for_status = lambda: None
    poll_failed.json = lambda: {"status": "failed", "error": "something went wrong"}
    mock_get.side_effect = [poll_failed]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "fail.mp4"
    with pytest.raises(RuntimeError, match="failed"):
        c.generate_video("bad prompt", image_path=None, save_path=str(out))


# ── test: no polling_url in submit response raises RuntimeError ───────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_missing_polling_url_raises(mock_post, mock_get, tmp_path):
    """Submit response without polling_url raises RuntimeError."""
    bad_submit = MagicMock(status_code=200)
    bad_submit.raise_for_status = lambda: None
    bad_submit.json = lambda: {"id": "job_no_polling_url"}  # old format — no polling_url
    mock_post.side_effect = [bad_submit]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0)
    out = tmp_path / "err.mp4"
    with pytest.raises(RuntimeError, match="polling_url"):
        c.generate_video("test", image_path=None, save_path=str(out))


# ── test: max_polls exhausted raises RuntimeError ────────────────────────────

@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_max_polls_exhausted_raises(mock_post, mock_get, tmp_path):
    """Exhausting max_polls without completion raises RuntimeError."""
    mock_post.side_effect = [_submit_post_mock()]
    pending = MagicMock(status_code=200)
    pending.raise_for_status = lambda: None
    pending.json = lambda: {"status": "pending"}
    # Return 3 pending polls with max_polls=3
    mock_get.side_effect = [pending, pending, pending]

    c = OpenRouterVideoClient(api_key="sk-test", poll_interval=0, max_polls=3)
    out = tmp_path / "timeout.mp4"
    with pytest.raises(RuntimeError):
        c.generate_video("test", image_path=None, save_path=str(out))

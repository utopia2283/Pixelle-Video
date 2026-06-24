# tests/api_services/test_video_openrouter.py
from unittest.mock import patch, MagicMock
from pixelle_video.services.api_services.video_openrouter import OpenRouterVideoClient


@patch("pixelle_video.services.api_services.video_openrouter.requests.get")
@patch("pixelle_video.services.api_services.video_openrouter.requests.post")
def test_generate_video_flow(mock_post, mock_get, tmp_path):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "job_1"})
    mock_post.return_value.raise_for_status = lambda: None
    poll = MagicMock(status_code=200, json=lambda: {"status": "completed", "output": {"url": "http://x/v.mp4"}})
    poll.raise_for_status = lambda: None
    dl = MagicMock(status_code=200, content=b"MP4DATA"); dl.raise_for_status = lambda: None
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

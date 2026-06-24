"""OpenRouter Videos API client (Seedance 2.0 etc.). submit -> poll -> download.

NOTE: Response field names below are based on OpenRouter documentation defaults
and have NOT been verified against a live API call. At deploy time, probe the
real API and correct field names if needed:
  - submit response: we expect `id` (fallback: `job_id`)
  - poll response: we expect `status` in ("completed", "succeeded")
  - video url: we expect `output.url` (fallback: `data.video.url`)
# field 名待 deploy 階段真 API 校正
"""

import os
import time
import base64
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OpenRouterVideoClient:
    """
    OpenRouter Videos API client for Seedance 2.0 and other video models.
    Uses async submit -> poll -> download flow (sync, caller wraps with asyncio.to_thread).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        timeout: int = 180,
        poll_interval: int = 5,
        max_polls: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.local_proxy = local_proxy
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_polls = max_polls

        if not self.api_key:
            logger.warning("OpenRouterVideoClient: OPENROUTER_API_KEY not set")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _proxies(self) -> Optional[dict]:
        if not self.local_proxy:
            return None
        return {"http": self.local_proxy, "https": self.local_proxy}

    @staticmethod
    def _frame(path: str, ftype: str) -> dict:
        """Encode a local image file as a data URI frame for first_frame/last_frame."""
        with open(path, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        ext = "png" if path.lower().endswith(".png") else "jpeg"
        return {"frame_type": ftype, "image_url": {"url": f"data:image/{ext};base64,{b}"}}

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str = "bytedance/seedance-2.0",
        duration: int = 5,
        last_image_path: Optional[str] = None,
        video_ratio: str = "16:9",
        resolution: str = "720p",
        **kwargs,
    ) -> str:
        """
        Full video generation flow: submit -> poll -> download.

        Args:
            prompt: Text prompt describing the video.
            image_path: Local path to the first-frame image (None for text-to-video).
            save_path: Local path where the downloaded video will be saved.
            model: OpenRouter model identifier. Default: bytedance/seedance-2.0.
            duration: Video duration in seconds.
            last_image_path: Optional local path to the last-frame image.
            video_ratio: Aspect ratio string, e.g. "16:9".
            resolution: Output resolution, e.g. "720p".
            **kwargs: Extra fields forwarded to the API payload.

        Returns:
            Remote video URL (video is also saved to save_path).
        """
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")

        # Build payload
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": video_ratio,  # field 名待 deploy 階段真 API 校正
            "resolution": resolution,
        }

        # Optional frame images (first / last frame)
        frames = []
        if image_path:
            frames.append(self._frame(image_path, "first_frame"))
        if last_image_path:
            frames.append(self._frame(last_image_path, "last_frame"))
        if frames:
            payload["frame_images"] = frames  # field 名待 deploy 階段真 API 校正

        # Forward any extra caller-supplied fields
        payload.update(kwargs)

        # ── Step 1: Submit ────────────────────────────────────────────────────
        logger.info(f"OpenRouterVideoClient: submitting job model={model}, duration={duration}s")
        r = requests.post(
            f"{self.base_url}/videos",  # POST /api/v1/videos — 待真 API 校正
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            proxies=self._proxies(),
        )
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id") or job.get("job_id")  # field 名待 deploy 階段真 API 校正
        if not job_id:
            raise RuntimeError(f"OpenRouter Videos API did not return a job id: {job}")
        logger.info(f"OpenRouterVideoClient: job submitted, id={job_id}")

        # ── Step 2: Poll ──────────────────────────────────────────────────────
        video_url = ""
        last_poll: dict = {}
        for i in range(self.max_polls):
            p = requests.get(
                f"{self.base_url}/videos/{job_id}",  # GET /api/v1/videos/{id} — 待真 API 校正
                headers=self._headers(),
                timeout=self.timeout,
                proxies=self._proxies(),
            )
            p.raise_for_status()
            last_poll = p.json()
            status = last_poll.get("status")

            if status in ("completed", "succeeded"):  # field 名待 deploy 階段真 API 校正
                # Try output.url first, then data.video.url
                video_url = (
                    (last_poll.get("output") or {}).get("url")  # field 名待 deploy 階段真 API 校正
                    or (last_poll.get("data") or {}).get("video", {}).get("url", "")
                )
                logger.info(f"OpenRouterVideoClient: job completed, url={video_url}")
                break

            if status in ("failed", "error", "expired"):
                raise RuntimeError(f"OpenRouter video job failed: {last_poll}")

            logger.debug(
                f"OpenRouterVideoClient: polling job {job_id}, status={status}, attempt={i + 1}"
            )
            if self.poll_interval:
                time.sleep(self.poll_interval)

        if not video_url:
            raise RuntimeError(
                f"OpenRouter video: no video URL after {self.max_polls} polls. last={last_poll}"
            )

        # ── Step 3: Download ──────────────────────────────────────────────────
        # Strategy: attempt download WITHOUT Authorization header first.
        # OpenRouter typically returns a pre-signed CDN URL; sending an Authorization
        # header to a CDN pre-signed URL can cause a 403 (the CDN treats extra auth
        # as a signature mismatch).  Only fall back to sending auth if the no-auth
        # attempt is rejected with 401 or 403 (in case the endpoint genuinely requires it).
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        dl = requests.get(
            video_url,
            headers=None,          # no auth — CDN pre-signed URL does not need it
            timeout=self.timeout,
            proxies=self._proxies(),
            stream=True,
        )
        if dl.status_code in (401, 403):
            # No-auth attempt was rejected: retry with Authorization header.
            # This handles the rare case where the download URL is an authenticated
            # endpoint rather than a CDN pre-signed URL.
            logger.debug(
                f"OpenRouterVideoClient: no-auth download got {dl.status_code}, retrying with Authorization header"
            )
            dl = requests.get(
                video_url,
                headers=self._headers(),   # retry with auth
                timeout=self.timeout,
                proxies=self._proxies(),
                stream=True,
            )
        dl.raise_for_status()

        # Stream to disk in 8 KiB chunks to avoid loading the entire video into RAM
        # (same pattern as video_seedance.py:_download_video).
        with open(save_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"OpenRouterVideoClient: video saved to {save_path}")

        return video_url

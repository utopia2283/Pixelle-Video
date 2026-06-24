"""OpenRouter Videos API client (Seedance 1.5 Pro etc.). submit -> poll -> download.

Frame images MUST be public HTTP URLs — data: URIs are silently ignored by Seedance.
Local frame images are uploaded to tmpfiles.org to obtain public URLs before submission.

Response field names verified against OpenRouter Videos API:
  - submit response: polling_url (direct poll URL returned by the server)
  - poll response: status in ("completed", "succeeded"), unsigned_urls[0] for download
"""

import os
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"


class OpenRouterVideoClient:
    """
    OpenRouter Videos API client for Seedance 1.5 Pro and other video models.
    Uses async submit -> poll -> download flow (sync, caller wraps with asyncio.to_thread).

    Frame images are uploaded to tmpfiles.org to obtain public URLs before submission,
    because Seedance ignores data: URI frames.
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
    def _tmpfiles_direct_url(url: str) -> str:
        """Convert a tmpfiles.org page URL to its direct-download URL.

        https://tmpfiles.org/12345/x.png  →  https://tmpfiles.org/dl/12345/x.png
        """
        prefix = "https://tmpfiles.org/"
        if url.startswith(prefix) and not url.startswith("https://tmpfiles.org/dl/"):
            return "https://tmpfiles.org/dl/" + url[len(prefix):]
        return url

    def _upload_frame(self, path: str) -> str:
        """Upload a local image file to tmpfiles.org and return its public direct-link URL.

        Uses a browser-like User-Agent to avoid bot-filter rejections.
        """
        logger.debug(f"OpenRouterVideoClient: uploading frame to tmpfiles.org: {path}")
        with open(path, "rb") as fh:
            resp = requests.post(
                TMPFILES_UPLOAD_URL,
                files={"file": fh},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
                proxies=self._proxies(),
            )
        resp.raise_for_status()
        resp_json = resp.json()
        raw_url = resp_json.get("data", {}).get("url")
        if not raw_url:
            raise RuntimeError(f"tmpfiles upload failed: {resp_json}")
        direct_url = self._tmpfiles_direct_url(raw_url)
        logger.debug(f"OpenRouterVideoClient: frame uploaded, direct_url={direct_url}")
        return direct_url

    def _frame(self, path: str, ftype: str) -> dict:
        """Upload a local image file to tmpfiles.org and return a frame_images entry."""
        public_url = self._upload_frame(path)
        return {"frame_type": ftype, "image_url": {"url": public_url}}

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str = "bytedance/seedance-1-5-pro",
        duration: int = 5,
        last_image_path: Optional[str] = None,
        video_ratio: str = "16:9",
        resolution: str = "720p",
        **kwargs,
    ) -> str:
        """
        Full video generation flow: upload frames → submit → poll → download.

        Args:
            prompt: Text prompt describing the video.
            image_path: Local path to the first-frame image (None for text-to-video).
            save_path: Local path where the downloaded video will be saved.
            model: OpenRouter model identifier. Default: bytedance/seedance-1-5-pro.
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
            "aspect_ratio": video_ratio,
            "resolution": resolution,
        }

        # Optional frame images (first / last frame) — upload to tmpfiles for public URLs.
        # Seedance ignores data: URI frames; public HTTP URLs are required.
        frames = []
        if image_path:
            frames.append(self._frame(image_path, "first_frame"))
        if last_image_path:
            frames.append(self._frame(last_image_path, "last_frame"))
        if frames:
            payload["frame_images"] = frames

        # Forward any extra caller-supplied fields
        payload.update(kwargs)

        # ── Step 1: Submit ────────────────────────────────────────────────────
        logger.info(f"OpenRouterVideoClient: submitting job model={model}, duration={duration}s")
        r = requests.post(
            f"{self.base_url}/videos",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            proxies=self._proxies(),
        )
        r.raise_for_status()
        job = r.json()
        polling_url = job.get("polling_url")
        if not polling_url:
            raise RuntimeError(
                f"OpenRouter Videos API did not return a polling_url in submit response: {job}"
            )
        logger.info(f"OpenRouterVideoClient: job submitted, polling_url={polling_url}")

        # ── Step 2: Poll ──────────────────────────────────────────────────────
        video_url = ""
        last_poll: dict = {}
        for i in range(self.max_polls):
            p = requests.get(
                polling_url,
                headers=self._headers(),
                timeout=self.timeout,
                proxies=self._proxies(),
            )
            p.raise_for_status()
            last_poll = p.json()
            status = last_poll.get("status")

            if status in ("completed", "succeeded"):
                unsigned_urls = last_poll.get("unsigned_urls") or []
                if not unsigned_urls:
                    raise RuntimeError("unsigned_urls empty on completion")
                video_url = unsigned_urls[0]
                logger.info(f"OpenRouterVideoClient: job completed, url={video_url}")
                break

            if status in ("failed", "error", "expired"):
                raise RuntimeError(f"OpenRouter video job failed: {last_poll}")

            logger.debug(
                f"OpenRouterVideoClient: polling {polling_url}, status={status}, attempt={i + 1}"
            )
            if self.poll_interval:
                time.sleep(self.poll_interval)

        if not video_url:
            raise RuntimeError(
                f"OpenRouter video: no video URL after {self.max_polls} polls. last={last_poll}"
            )

        # ── Step 3: Download ──────────────────────────────────────────────────
        # unsigned_urls are authenticated endpoints — always send Authorization header.
        # (Unlike CDN pre-signed URLs, unsigned_urls require Bearer auth from OpenRouter.)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        dl = requests.get(
            video_url,
            headers=self._headers(),   # Bearer auth required for unsigned_urls
            timeout=self.timeout,
            proxies=self._proxies(),
            stream=True,
        )
        dl.raise_for_status()

        # Stream to disk in 8 KiB chunks to avoid loading the entire video into RAM.
        with open(save_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"OpenRouterVideoClient: video saved to {save_path}")

        return video_url

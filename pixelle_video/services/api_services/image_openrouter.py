"""OpenRouter image generation via Grok chat/completions (modalities: image)."""
import os
import base64
import uuid
import logging
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


class OpenRouterImageClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.local_proxy = local_proxy
        self.timeout = timeout
        if not self.api_key:
            logger.warning("OpenRouterImageClient: OPENROUTER_API_KEY 未設置")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _proxies(self):
        return (
            {"http": self.local_proxy, "https": self.local_proxy}
            if self.local_proxy
            else None
        )

    @staticmethod
    def _to_data_uri(path: str) -> str:
        with open(path, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        ext = "png" if path.lower().endswith(".png") else "jpeg"
        return f"data:image/{ext};base64,{b}"

    @staticmethod
    def _url_to_bytes(url: str) -> bytes:
        """Download an image URL or decode a data URI to raw bytes."""
        if url.startswith("data:"):
            # data:image/png;base64,<...>
            _, encoded = url.split(",", 1)
            return base64.b64decode(encoded)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    def generate_image(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        model: str = "x-ai/grok-imagine-image-quality",
        save_dir: Optional[str] = None,
        video_ratio: str = "16:9",
        resolution: str = "2K",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Generate an image via Grok chat/completions.

        ``video_ratio`` and ``resolution`` are accepted for API compatibility with
        downstream callers but are not forwarded to the chat/completions endpoint
        (Grok ignores them there).
        """
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")

        # Build content array
        content: list = [{"type": "text", "text": prompt}]
        if image_paths:
            for p in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_data_uri(p)},
                    }
                )

        payload = {
            "model": model,
            # IMPORTANT: modalities must be exactly ["image"] — adding "text" causes 404
            "modalities": ["image"],
            "messages": [{"role": "user", "content": content}],
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            proxies=self._proxies(),
        )
        resp.raise_for_status()

        # Extract image URL from response
        # Shape: choices[0].message.images[0].image_url.url  (primary)
        #        choices[0].message.images[0].url             (fallback)
        data = resp.json()
        try:
            images = data["choices"][0]["message"]["images"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"OpenRouter Grok image: unexpected response shape. resp={resp.text[:300]}"
            ) from exc

        save_dir = save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        paths: List[str] = []

        for item in images:
            # primary path
            url = item.get("image_url", {}).get("url") or item.get("url", "")
            if not url:
                logger.warning("OpenRouterImageClient: image item has no url, skipping: %s", item)
                continue
            raw = self._url_to_bytes(url)
            out = os.path.join(save_dir, f"or_{uuid.uuid4().hex[:8]}.png")
            with open(out, "wb") as f:
                f.write(raw)
            paths.append(out)

        if not paths:
            raise RuntimeError(
                f"OpenRouter Grok image: no images in response. resp={resp.text[:300]}"
            )
        return paths

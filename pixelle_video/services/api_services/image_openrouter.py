"""OpenRouter Images API client (Seedream 4.5 etc.)."""
import os, base64, uuid, logging
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


class OpenRouterImageClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 local_proxy: Optional[str] = None, timeout: int = 180) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL")
                         or "https://openrouter.ai/api/v1").rstrip("/")
        self.local_proxy = local_proxy
        self.timeout = timeout
        if not self.api_key:
            logger.warning("OpenRouterImageClient: OPENROUTER_API_KEY 未設置")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _proxies(self):
        return {"http": self.local_proxy, "https": self.local_proxy} if self.local_proxy else None

    @staticmethod
    def _to_data_uri(path: str) -> str:
        with open(path, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        ext = "png" if path.lower().endswith(".png") else "jpeg"
        return f"data:image/{ext};base64,{b}"

    def generate_image(self, prompt: str, image_paths: Optional[List[str]] = None,
                       model: str = "bytedance-seed/seedream-4.5", save_dir: Optional[str] = None,
                       video_ratio: str = "16:9", resolution: str = "2K",
                       session_id: Optional[str] = None) -> List[str]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        payload = {"model": model, "prompt": prompt, "resolution": resolution, "aspect_ratio": video_ratio}
        if image_paths:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": self._to_data_uri(p)}} for p in image_paths
            ]
        resp = requests.post(f"{self.base_url}/images", headers=self._headers(),
                             json=payload, timeout=self.timeout, proxies=self._proxies())
        resp.raise_for_status()
        data = resp.json().get("data", [])
        save_dir = save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        paths: List[str] = []
        for item in data:
            b64 = item.get("b64_json") or item.get("image_url", {}).get("url", "")
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            out = os.path.join(save_dir, f"or_{uuid.uuid4().hex[:8]}.png")
            with open(out, "wb") as f:
                f.write(base64.b64decode(b64))
            paths.append(out)
        if not paths:
            raise RuntimeError(f"OpenRouter image: empty data. resp={resp.text[:300]}")
        return paths

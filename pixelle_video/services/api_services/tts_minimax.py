"""MiniMax T2A (speech-2.8-turbo) TTS client."""
import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class MiniMaxTTSClient:
    """Client for MiniMax Text-to-Audio v2 API (speech-2.8-turbo)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        group_id: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = (
            base_url or os.getenv("MINIMAX_BASE_URL") or "https://api.minimax.io/v1"
        ).rstrip("/")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID")
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str,
        speed: float = 1.0,
        model: str = "speech-2.8-turbo",
    ) -> str:
        """
        Synthesize speech and write the result to output_path as an mp3 file.

        Args:
            text: Text to convert to speech.
            output_path: Destination file path (will be created if needed).
            voice: MiniMax voice_id (e.g. "Cantonese_GentleLady").
            speed: Speech speed multiplier (1.0 = normal).
            model: MiniMax TTS model name.

        Returns:
            The absolute output_path string on success.
        """
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY not set.")

        url = f"{self.base_url}/t2a_v2"
        if self.group_id:
            url += f"?GroupId={self.group_id}"

        payload = {
            "model": model,
            "text": text,
            "stream": False,
            "language_boost": "Chinese,Yue",
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice,
                "speed": speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "format": "mp3",
                "sample_rate": 32000,
                "bitrate": 128000,
                "channel": 1,
            },
        }

        logger.info("MiniMax TTS synthesize: voice=%s, model=%s", voice, model)
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        r.raise_for_status()

        body = r.json()
        base_resp = body.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax TTS error: {base_resp}")

        audio_hex = body.get("data", {}).get("audio", "")
        if not audio_hex:
            raise RuntimeError(f"MiniMax TTS: empty audio in response. {base_resp}")

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(bytes.fromhex(audio_hex))

        logger.info("MiniMax TTS written to %s", output_path)
        return output_path

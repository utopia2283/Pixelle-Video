"""Integration: verify the OpenRouter+MiniMax stack config flows from
PixelleVideoConfig.to_dict() through to the services (not stripped by schema)."""
from pixelle_video.config.schema import PixelleVideoConfig
from pixelle_video.services.tts_service import TTSService


def test_tts_minimax_config_survives_to_dict_and_reaches_service():
    cfg = PixelleVideoConfig(**{
        "comfyui": {
            "tts": {
                "inference_mode": "minimax",
                "minimax": {
                    "model": "speech-2.8-turbo",
                    "voice": "Cantonese_GentleLady",
                    "speed": 1.1,
                },
            }
        }
    })
    d = cfg.to_dict()
    # schema 唔 strip minimax
    assert d["comfyui"]["tts"]["inference_mode"] == "minimax"
    assert d["comfyui"]["tts"]["minimax"]["voice"] == "Cantonese_GentleLady"
    assert d["comfyui"]["tts"]["minimax"]["speed"] == 1.1
    # ComfyBaseService scope self.config 到 comfyui.tts，service 收到 minimax
    svc = TTSService(d, core=None)
    assert svc.config["inference_mode"] == "minimax"
    assert svc.config["minimax"]["voice"] == "Cantonese_GentleLady"


def test_minimax_defaults_when_omitted():
    cfg = PixelleVideoConfig()
    d = cfg.to_dict()
    mm = d["comfyui"]["tts"]["minimax"]
    assert mm["model"] == "speech-2.8-turbo"
    assert mm["voice"] == "Cantonese_GentleLady"


def test_openrouter_provider_present_in_schema():
    cfg = PixelleVideoConfig(**{"api_providers": {"openrouter": {"api_key": "x"}}})
    d = cfg.to_dict()
    assert d["api_providers"]["openrouter"]["api_key"] == "x"


def test_tts_minimax_config_api_key_group_id_survive_schema():
    """INT-1: api_key + group_id must NOT be stripped by TTSMiniMaxConfig pydantic schema."""
    cfg = PixelleVideoConfig(**{
        "comfyui": {
            "tts": {
                "inference_mode": "minimax",
                "minimax": {
                    "model": "speech-2.8-turbo",
                    "voice": "Cantonese_GentleLady",
                    "speed": 1.0,
                    "api_key": "sk-mm-secret",
                    "group_id": "grp-789",
                },
            }
        }
    })
    d = cfg.to_dict()
    mm = d["comfyui"]["tts"]["minimax"]
    assert mm.get("api_key") == "sk-mm-secret", (
        "api_key was stripped by TTSMiniMaxConfig — add api_key field to schema"
    )
    assert mm.get("group_id") == "grp-789", (
        "group_id was stripped by TTSMiniMaxConfig — add group_id field to schema"
    )


def test_tts_minimax_empty_api_key_does_not_override_env(monkeypatch):
    """Empty-string api_key/group_id in config must not shadow the env fallback.

    When the schema field defaults to "" and the caller did NOT supply a value,
    the effective key passed to MiniMaxTTSClient must be None (not ""), so the
    client can fall back to MINIMAX_API_KEY / MINIMAX_GROUP_ID env vars.
    """
    import asyncio
    from unittest.mock import patch, MagicMock
    from pixelle_video.services.tts_service import TTSService

    monkeypatch.setenv("MINIMAX_API_KEY", "env-key-123")

    config = {
        "comfyui": {
            "tts": {
                "inference_mode": "minimax",
                "minimax": {
                    # api_key / group_id deliberately absent — schema default is ""
                    "voice": "Cantonese_GentleLady",
                    "speed": 1.0,
                    "model": "speech-2.8-turbo",
                },
            }
        },
    }

    with patch("pixelle_video.services.tts_service.MiniMaxTTSClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.synthesize.return_value = "output/fake.mp3"
        MockClient.return_value = mock_instance

        svc = TTSService(config=config, core=None)
        asyncio.get_event_loop().run_until_complete(svc("hi"))

        # api_key passed to constructor must be None (falsy) — NOT ""
        ctor_kwargs = MockClient.call_args
        api_key_arg = ctor_kwargs.kwargs.get("api_key") if ctor_kwargs.kwargs else None
        assert api_key_arg is None, (
            f"Empty-string api_key '{api_key_arg}' must not be passed to MiniMaxTTSClient "
            "— use `mm.get('api_key') or None` to avoid shadowing env var"
        )

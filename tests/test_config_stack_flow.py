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

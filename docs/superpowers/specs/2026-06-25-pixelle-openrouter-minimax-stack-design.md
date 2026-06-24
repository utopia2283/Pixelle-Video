# Pixelle-Video × OpenRouter + MiniMax Stack — Design Spec

- 日期: 2026-06-25
- 狀態: Draft (待 user review)
- 目標機: office4090 (Linux, 香港)，公開 URL `https://pv.syncbuddy.ai`
- 上游: `AIDC-AI/Pixelle-Video` (Apache-2.0)，本地已 shallow clone

## 1. 目標

將開源 Pixelle-Video（一句主題 → 自動出片 pipeline）部署上 4090，公開 `pv.syncbuddy.ai`，
並將其媒體生成 stack 換成用戶已驗過嘅一套：

| 能力 | Provider / Endpoint | Model | Key (來自 `~/.hermes/.env`) |
|------|--------------------|-------|------------------------------|
| LLM（寫稿/分鏡） | MiniMax OpenAI-compat `api.minimax.io/v1/chat/completions` | `MiniMax-M3` | `MINIMAX_API_KEY` |
| 出圖 | OpenRouter Images `POST /api/v1/images` | `bytedance-seed/seedream-4.5` | `OPENROUTER_API_KEY` |
| 出片 | OpenRouter Videos `POST /api/v1/videos`（submit→poll→download） | `bytedance/seedance-2.0` | `OPENROUTER_API_KEY` |
| TTS | MiniMax `POST https://api.minimax.io/v1/t2a_v2` | `speech-2.8-turbo`（廣東話，已驗 2026-06-06） | `MINIMAX_API_KEY` |

非目標（YAGNI）：
- 唔裝本地 ComfyUI（GPU 推理一律走 cloud API）。
- 唔上 Vercel（Streamlit 係長駐 server，Vercel 無 GPU 無長駐，做唔到 host）。
- 唔上 Docker（沿用 4090 既有 systemd --user + uv 套路，同 opc/df 一致）。
- `api/app.py`（FastAPI/MCP）唔上線，只跑 `web/app.py`（Streamlit UI）。

## 2. 現狀 / 架構認知（已讀 repo 確認）

- 兩個 service：`web/app.py`（Streamlit :8501）+ `api/app.py`（FastAPI :8000）。出街只需 `web`。
- **媒體生成核心係 ComfyUI workflow-based**：`pixelle_video/services/media.py`、`tts_service.py`
  都 extends `comfy_base_service.ComfyBaseService`，經 ComfyKit 執行 `workflows/{runninghub,selfhost}/*.json`。
  config 揀嘅係 `comfyui.image.default_workflow` / `video.default_workflow` / `tts.default_workflow`（workflow 檔路徑），
  schema 入面**冇 provider 選擇欄位**。
- `pixelle_video/services/api_services/`（`image_client.py` / `video_client.py` / `image_seedream.py` /
  `video_seedance.py` 等）係**另一條獨立 direct-API 路**，by model name route，行字節 **ARK**
  (`ark.cn-beijing.volces.com/api/v3`)，**唔係主 pipeline 出圖出片嗰條路**（主 pipeline 行 ComfyUI workflow）。
- LLM 走 `pixelle_video/services/llm_service.py`，OpenAI SDK compatible（填 `llm.base_url/api_key/model` 即用）。
- 主 pipeline：`pixelle_video/pipelines/standard.py`，用 `media_workflow` / `tts_workflow` 參數驅動 workflow。

關鍵 implication：要令出圖/出片行 OpenRouter direct API（而唔係 ComfyUI workflow），
**必須喺媒體 pipeline 加一條 direct-provider dispatch**，唔係淨係加 client class。

## 3. 架構決定

### 3.1 媒體生成：加「direct provider」模式

喺 config 新增一個 provider 選擇維度（沿用既有 `api_providers` 結構，新增 `openrouter` provider），
並喺主 pipeline 的出圖/出片步驟加 dispatch：

- 當選定 OpenRouter direct provider → 行新 adapter（呢個 spec 嘅 3 個）。
- 否則 → 行原本 ComfyUI workflow（保留，唔破壞上游行為）。

dispatch 點：`standard.py`（同 `custom.py` / `linear.py` 如共用同一媒體呼叫）出圖/出片嗰兩步，
抽一個薄 `MediaProvider` 介面，workflow 路同 direct 路各實作一份。實作細節落 plan。

### 3.2 LLM：純 config，零 code

`config.yaml`:
```yaml
llm:
  api_key: ${MINIMAX_API_KEY}
  base_url: "https://api.minimax.io/v1"
  model: "MiniMax-M3"
```
（key 由環境注入，唔硬寫入 repo。注入機制落 plan：service 啟動時由 `~/.hermes/.env` export。）

### 3.3 三個新 adapter

A. **出圖 `image_openrouter.py`**
- `POST https://openrouter.ai/api/v1/images`，body `{model, prompt, resolution, aspect_ratio}`；
  圖生圖加 `input_references:[{type:"image_url", image_url:{url}}]`。
- 回傳 `result["data"][0]["b64_json"]`（base64）→ 解碼存檔。
- 插入點：媒體 direct-provider dispatch（見 3.1），同時可註冊入 `image_client.py` 方便 api 路共用。

B. **出片 OpenRouter Seedance video client**
- `POST https://openrouter.ai/api/v1/videos` submit → 攞 job id + polling_url → poll → 下載 `unsigned_urls[0]`（帶 Bearer）。
- 支持 first/last frame（圖生片補間）+ reference-to-video。
- 插入點：媒體 direct-provider dispatch（見 3.1），可與既有 `video_client.py` 並列。

C. **TTS MiniMax adapter**
- `POST https://api.minimax.io/v1/t2a_v2`，model `speech-2.8-turbo`，廣東話 voice。
- 接入 `tts_service.py`：`__call__` 現支援 local edge-tts 與 ComfyUI 兩 mode，新增第三 mode `minimax`，
  令 pipeline 選 minimax 時行新 adapter（不經 ComfyKit）。
- 回傳 audio bytes → 存檔交畀 pipeline 合成。

## 4. 部署設計（沿用 opc/df 套路）

1. **GitHub**：fork `AIDC-AI/Pixelle-Video` → 用戶 account；4090 clone fork 去 `~/pixelle-video`。
2. **依賴**：`uv sync` + `uv run playwright install`（templates 用 playwright 截圖）+ 確認系統 `ffmpeg`。
3. **config**：`config.yaml`（由 `config.example.yaml` 起手），填 §3 stack；key 由 `~/.hermes/.env` 注入。
4. **service**：`systemd --user` `pixelle-video.service`，`ExecStart=… streamlit run web/app.py
   --server.port 8501 --server.address 127.0.0.1`，`Restart=always`，開機自起。
5. **公開**：cloudflared `~/.cloudflared/config.yml` ingress 加 `pv.syncbuddy.ai → http://localhost:8501`
   （tunnel `63fc0a1f` = syncbuddy-edu）；syncbuddy.ai zone（CF account Match2289@gmail.com）起 `pv` CNAME
   （proxied）→ `63fc0a1f-…cfargotunnel.com`。Streamlit WebSocket（`_stcore/stream`）經 cloudflared 支援。
6. **Streamlit over proxy**：設 `server.enableXsrfProtection=false` / 適當 forwarded headers / `STREAMLIT_SERVER_HEADLESS=true`，
   確保經 subdomain 反代 OK（落 plan 試）。

## 5. 風險 / Open items（落 plan/impl 解決，唔卡 design）

- MiniMax T2A 回傳格式（hex/base64/url）+ 要唔要 `GroupId`；廣東話 voice id。
- Seedance 2.0 嘅 first/last frame 參數點對接 Pixelle pipeline（佢預設係單圖→片定 t2v）。
- 媒體 dispatch 抽象唔好破壞上游 RunningHub/selfhost workflow 路（保留可回退）。
- OpenRouter Images `supported_parameters` 對 seedream-4.5 只列 `resolution`/`seed`，aspect_ratio 要實測 clamp。
- key 注入：唔好將 secret commit 入 repo；用 systemd `EnvironmentFile` 指 `~/.hermes/.env` 或專屬 env。

## 6. 驗收

1. `https://pv.syncbuddy.ai` 開到 Streamlit UI。
2. 入一條 topic → LLM(MiniMax-M3) 出稿 → 出圖(seedream-4.5) → 出片(seedance-2.0) → TTS(minimax 粵語) → 合成成片，端到端成功。
3. service 重啟 / 4090 重開機後自動返。
4. 上游 ComfyUI workflow 路仍可用（回退唔破壞）。

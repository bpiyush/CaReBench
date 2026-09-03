# Text-to-Video Search Demo

FastAPI + light minimalist UI (no Gradio). Two pages: **prepare** → **search**.

## On `dev1`

```bash
cd /users/piyush/projects/CaReBench/tara-presentation
./remote_start_backend.sh --port 7860   # picks free GPUs, backgrounds server
# stop:
./remote_stop_backend.sh 7860
```

## On your laptop

```bash
ssh -N -L 7860:localhost:7860 dev1
```

Open **http://localhost:7860**

## Features

- **Sample %** — randomly encode a subset (default 10%) while debugging
- **Cache** — `/work/piyush/experiments/TARA-demo/cache/`  
  - `embeddings/{model}/{dataset}.pt` — dict `video_id → embedding`  
  - `previews/` — downsized display clips

- Videos served via `/api/video/{id}` (real `<video>` tags)

## Flow

1. Config page: model / dataset / split / sample % → **Next**
2. Coffee + progress + ETA while embedding
3. Search page: left sidebar (active setup), light search box, gray example chips, video grid with scores

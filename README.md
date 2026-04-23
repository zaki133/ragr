# Nextcloud RAG bundle using llama.cpp

This bundle is preconfigured for:

- **Model:** `Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf`
- **Runtime:** `llama-server` from `llama.cpp`
- **RAG source:** Nextcloud WebDAV folder `/model_data`
- **UI:** Gradio on port `7860`

## What it includes

- Nextcloud WebDAV sync from `/model_data`
- local text extraction for PDF, TXT, MD, DOCX, CSV
- FAISS index with sentence-transformer embeddings
- OpenAI-compatible generation via `llama-server`
- simple Gradio web UI on port `7860`
- systemd units for sync timer, UI, and llama.cpp server

## Install

```bash
cd llamacpp_nextcloud_rag_bundle_llama32_3b
chmod +x install.sh
./install.sh
```

## Configure

```bash
cp /opt/nextcloud-rag/.env.example /opt/nextcloud-rag/.env
nano /opt/nextcloud-rag/.env
```

Fill in at least:

- `NEXTCLOUD_WEBDAV_URL`
- `NEXTCLOUD_USERNAME`
- `NEXTCLOUD_APP_PASSWORD`

The rest is already set for the selected model.

## Put the model in place

Expected model path:

```bash
/opt/models/Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf
```

Create the folder if needed:

```bash
mkdir -p /opt/models
ls -lh /opt/models
```

## Important GPU note

The default config uses:

- `LLAMA_CTX_SIZE=2048`
- `LLAMA_N_GPU_LAYERS=0`

That is the safest default because it starts even without GPU access.

If this machine has access to your GTX 1060 3GB, edit `/opt/nextcloud-rag/.env` and try:

```env
LLAMA_N_GPU_LAYERS=20
```

If the server fails to start, reduce it to `15`, then `10`, then `0`.

## Start llama.cpp manually

```bash
/opt/nextcloud-rag/start_llama_server.sh
```

API endpoint:

```text
http://127.0.0.1:8080/v1
```

## First sync and index build

```bash
source /opt/nextcloud-rag/.venv/bin/activate
python /opt/nextcloud-rag/rag_app.py sync-index
```

## Test a question

```bash
python /opt/nextcloud-rag/rag_app.py ask "What files are in the knowledge base?"
```

## Start the web UI manually

```bash
python /opt/nextcloud-rag/rag_app.py serve --host 0.0.0.0 --port 7860
```

## Install systemd services

```bash
cp llama-server.service /etc/systemd/system/
cp nextcloud-rag-sync.service /etc/systemd/system/
cp nextcloud-rag-sync.timer /etc/systemd/system/
cp nextcloud-rag-ui.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now llama-server.service
systemctl enable --now nextcloud-rag-sync.timer
systemctl enable --now nextcloud-rag-ui.service
```

## Useful checks

```bash
systemctl status llama-server.service
systemctl status nextcloud-rag-ui.service
systemctl status nextcloud-rag-sync.timer
journalctl -u llama-server.service -n 100 --no-pager
journalctl -u nextcloud-rag-ui.service -n 100 --no-pager
journalctl -u nextcloud-rag-sync.service -n 100 --no-pager
```

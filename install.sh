#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/nextcloud-rag
LLAMA_DIR=/opt/llama.cpp
MODELS_DIR=/opt/models

apt update
apt install -y python3 python3-venv python3-pip build-essential cmake git curl wget

mkdir -p "$APP_DIR" "$MODELS_DIR"
cp rag_app.py "$APP_DIR/rag_app.py"
cp .env.example "$APP_DIR/.env.example"
cp README.md "$APP_DIR/README.md"
cp start_llama_server.sh "$APP_DIR/start_llama_server.sh"
chmod +x "$APP_DIR/start_llama_server.sh"

python3 -m venv "$APP_DIR/.venv"
source "$APP_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install requests python-dotenv gradio langchain langchain-community sentence-transformers faiss-cpu pypdf python-docx pandas openpyxl

action_msg=""
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  action_msg="cloned"
else
  action_msg="updated"
fi
cd "$LLAMA_DIR"
git pull --ff-only || true
cmake -B build
cmake --build build -j"$(nproc)"

echo
printf '%s\n' "Install complete. llama.cpp ${action_msg} in $LLAMA_DIR"
printf '%s\n' "Next steps:"
printf '%s\n' "1. cp $APP_DIR/.env.example $APP_DIR/.env"
printf '%s\n' "2. edit $APP_DIR/.env"
printf '%s\n' "3. download a GGUF model into $MODELS_DIR"
printf '%s\n' "4. optionally install the included systemd service files"

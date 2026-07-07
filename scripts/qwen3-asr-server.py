"""Local Qwen3-ASR server for Meetily.

OpenAI-compatible endpoint the app's qwen3Asr provider talks to:
    POST /v1/audio/transcriptions   (multipart: file=<wav>, language=<name, optional>)
    GET  /health

Run inside a venv that has: torch (CUDA), qwen-asr, fastapi, uvicorn, python-multipart.
    python qwen3-asr-server.py [--port 8585] [--model Qwen/Qwen3-ASR-1.7B]
"""
import argparse
import io
import tempfile

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from qwen_asr import Qwen3ASRModel

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8585)
parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
args = parser.parse_args()

print(f"Loading {args.model}...")
model = Qwen3ASRModel.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    max_new_tokens=512,
)
print("Model loaded.")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model": args.model}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: str = Form(None)):
    data = await file.read()
    # qwen_asr accepts a file path; keep it simple and robust across formats
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    results = model.transcribe(audio=path, language=language or None)
    return {"text": results[0].text, "language": results[0].language}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port)

# Stub Manifest Format

Stub manifests let the repo reference a large model file without committing the
file itself.

Each stub is a JSON document. Two provider shapes are supported.

**Google Drive** (Piper ONNX and other sub-5 GB assets):

```json
{
  "name": "en_US-lessac-medium.onnx",
  "backend": "piper",
  "target_path": "models/piper/en_US-lessac-medium.onnx",
  "source": {
    "provider": "google_drive",
    "file_id": "REPLACE_WITH_DRIVE_FILE_ID"
  },
  "sha256": "optional-checksum"
}
```

**HuggingFace** (large GGUF / quantized LLMs):

```json
{
  "name": "Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf",
  "backend": "llm",
  "target_path": "models/llm/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf",
  "source": {
    "provider": "huggingface",
    "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
    "filename": "Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf"
  }
}
```

Fields:

- `name`: display name for logs
- `backend`: logical consumer — `piper`, `whisper`, or `llm`
- `target_path`: local file path to materialize
- `source.provider`: `google_drive` or `huggingface`
- `source.file_id`: Drive file id (google_drive only)
- `source.repo`: HuggingFace `owner/repo` slug (huggingface only)
- `source.filename`: filename within the HF repo (huggingface only)
- `sha256`: optional checksum verification after download

Hydrate any stub with:

```bash
python scripts/fetch_stub.py <path-to-stub.json>
```

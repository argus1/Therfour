# Model Storage

This directory is the tracked home for local model artifacts used by Therfour.

Layout:

- `models/stubs/`: lightweight metadata-only manifests for Drive-backed assets
- `models/piper/`: Piper voice files used by `PIPER_MODEL_PATH`
- `models/llm/`: optional local LLM packs such as `.gguf` or `.safetensors`

Rules:

- Keep stub manifests in git.
- Keep large binary artifacts under this directory so `.gitattributes` can route
  them through Git LFS.
- Do not put ad hoc downloads in repo root.
- Use `models/.downloads/` only as a temporary staging directory.

Example:

```bash
python scripts/fetch_gdrive_stub.py models/stubs/piper/en_US-lessac-medium.onnx.stub.json
```

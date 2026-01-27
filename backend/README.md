# Telescope Backend

Deep learning model weight/activation inspector for Mamba-transformer hybrid models.

## Setup

```bash
# Install dependencies
uv sync

# Install PyTorch with CUDA support
uv add torch --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm (requires special flags)
uv pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

## Running

```bash
# Development server with auto-reload
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── api/              # API endpoints
│   │   ├── models.py     # Model management
│   │   ├── graph.py      # Graph extraction
│   │   ├── weights.py    # Weight inspection
│   │   └── activations.py # Activation capture
│   ├── core/             # Core functionality
│   │   ├── model_wrapper.py    # Model loading & hooks
│   │   ├── model_registry.py   # Model configurations
│   │   ├── graph_extractor.py  # FX graph extraction
│   │   └── distribution.py     # Stats computation
│   └── schemas/          # Pydantic models
└── pyproject.toml        # Project config
```

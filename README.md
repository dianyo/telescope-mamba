# Telescope Mamba

Deep learning model weight/activation inspector for Mamba-transformer hybrid models.

A tool for investigating model internals to better design algorithms for quantization, pruning, and model analysis.

## Target Models

- Nemotron 3 Nano (BF16 and FP8)
- MaTVLM
- TinyViM

## Architecture

- **Backend**: FastAPI (Python) - Model loading, weight/activation inspection, graph extraction
- **Frontend**: React + React Flow - Interactive model graph visualization, distribution plots

## Quick Start

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Features (Planned)

- [x] Backend project structure
- [ ] Model registry for Mamba-transformer hybrids
- [ ] Weight distribution visualization (histograms, heatmaps)
- [ ] Activation capture during evaluation
- [ ] Interactive architecture graph (React Flow)
- [ ] Quantization sensitivity analysis
- [ ] Per-channel outlier detection

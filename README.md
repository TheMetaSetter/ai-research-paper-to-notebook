# Paper-to-Notebook

A pedagogy-first pipeline for converting an AI research paper into an interactive Jupyter notebook that teaches the paper from first principles.

The current sample paper is:

`examples/sample_papers/saveliev-2025-human-guided-data-centric-llm-copilots.pdf`

Use this with:

`examples/saveliev_2025_first_time_reader_profile.json`

## Installation

This project uses `uv` for fast, reliable Python package management. If you don't have `uv` installed, install it from [astral.sh/uv](https://astral.sh/uv).

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-research-paper-to-notebook
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```
   This creates a virtual environment and installs all runtime dependencies.

3. **Install with dev tools:**
   ```bash
   uv sync --all-extras
   ```
   Include this if you're contributing or running tests. This adds pytest, ruff, mypy, and dvc.

### Running the Pipeline

#### Prerequisites: Update llama-server

Before starting the llama-server, ensure you have the latest version installed:

```bash
brew upgrade llama-server
```

Or follow the [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) for the latest version.

#### Run the Pipeline

Once installed, inspect the CLI:

```bash
uv run python src/main.py --help
```

To run the sample paper end to end, first start the local `llama.cpp` server in another terminal:

```bash
llama-server \
  --model checkpoints/gemma-4-E2B-it-Q4_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --threads 6 \
  --threads-batch 6 \
  --flash-attn auto \
  --jinja
```

Then run:

```bash
uv run python src/main.py run \
  examples/sample_papers/saveliev-2025-human-guided-data-centric-llm-copilots.pdf \
  --params-path params.yaml \
  --learner-profile examples/saveliev_2025_first_time_reader_profile.json
```

The output is written under:

```text
runs/saveliev-2025-human-guided-data-centric-llm-copilots/<run_id>/
```

The final notebook is:

```text
runs/.../notebook/final_notebook.ipynb
```

The validation report is:

```text
runs/.../validation_report/validation_report.json
```

### Development

Run tests with:
```bash
uv run pytest
```

Format and lint code:
```bash
uv run ruff check .
uv run ruff format src/
```

## Project Structure

- `src/` – Core pipeline implementation  
- `tests/` – Pytest test suite  
- `documents/` – Architecture and design documentation  
- `examples/` – Sample inputs and configurations  
- `prompts/` – Staged LLM prompts for notebook generation  
- `runs/` – Generated artifacts and logs  

See [documents/DESIGN_DOC.md](documents/DESIGN_DOC.md) for architectural details.

## License

See [LICENSE](LICENSE) for terms.

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unsloth/gemma-4-E2B-it-GGUF",
    local_dir="checkpoints",
    allow_patterns=["*Q4_0*"],
)
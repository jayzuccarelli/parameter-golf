"""
Run Parameter Golf training on Modal cloud GPUs.

Setup (run once):
    pip install modal
    modal setup

Download the dataset (run once, stored in a persistent Modal volume):
    modal run modal_run.py::download_data

    For a smaller subset while iterating:
    modal run modal_run.py::download_data --train-shards 1

Train on 1xH100 (for experimentation):
    modal run modal_run.py

Train on 8xH100 (for leaderboard submissions):
    modal run modal_run.py --gpus 8

Customize the run ID or dataset variant:
    modal run modal_run.py --gpus 8 --run-id my_experiment --variant sp1024
"""

import os
import subprocess
import sys

import modal

app = modal.App("parameter-golf")

# Persistent volume for the FineWeb dataset and tokenizer files.
# Created automatically on first use; reused on all subsequent runs.
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

# Container image: Python 3.11 + all training dependencies.
# Local code is baked into the image; Modal detects changes via content
# hashing and rebuilds automatically when files change.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "huggingface-hub",
        "datasets",
        "tiktoken",
        "sentencepiece",
        "kernels",
        "typing-extensions==4.15.0",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace/parameter-golf",
        # Exclude large local data files and build artifacts from the image.
        ignore=[".git", "data/datasets", "data/tokenizers", "__pycache__", "*.pyc"],
    )
)

WORKDIR = "/workspace/parameter-golf"
# Volume is mounted at the data subdirectory so downloads persist between runs.
VOLUMES = {f"{WORKDIR}/data": data_vol}


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=4,
    timeout=7200,
)
def download_data(variant: str = "sp1024", train_shards: int = 80):
    """Download FineWeb dataset to the persistent Modal volume.

    Run this once before training. Subsequent runs reuse the cached data.
    Pass --train-shards 1 for a much smaller download while iterating.
    """
    os.chdir(WORKDIR)
    subprocess.run(
        [
            sys.executable,
            "data/cached_challenge_fineweb.py",
            "--variant", variant,
            "--train-shards", str(train_shards),
        ],
        check=True,
    )
    # Flush volume writes so subsequent runs see the new files.
    data_vol.commit()


@app.function(
    image=image,
    gpu=modal.gpu.H100(count=1),
    volumes=VOLUMES,
    timeout=3600,
)
def train_1gpu(run_id: str = "baseline_sp1024", variant: str = "sp1024"):
    """Train on a single H100 GPU. Use this for experimentation."""
    os.chdir(WORKDIR)
    env = {
        **os.environ,
        "RUN_ID": run_id,
        "DATA_PATH": f"./data/datasets/fineweb10B_{variant}/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    }
    subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=env,
        check=True,
    )


@app.function(
    image=image,
    gpu=modal.gpu.H100(count=8),
    volumes=VOLUMES,
    timeout=3600,
)
def train_8gpu(run_id: str = "baseline_sp1024", variant: str = "sp1024"):
    """Train on 8xH100 GPUs. Required configuration for leaderboard submissions."""
    os.chdir(WORKDIR)
    env = {
        **os.environ,
        "RUN_ID": run_id,
        "DATA_PATH": f"./data/datasets/fineweb10B_{variant}/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    }
    subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        env=env,
        check=True,
    )


@app.local_entrypoint()
def main(
    gpus: int = 1,
    run_id: str = "baseline_sp1024",
    variant: str = "sp1024",
):
    """Launch a Parameter Golf training run on Modal.

    Args:
        gpus: Number of H100 GPUs to use (1 or 8). Default: 1.
        run_id: Identifier for this training run. Default: baseline_sp1024.
        variant: Dataset/tokenizer variant to use. Default: sp1024.
    """
    if gpus == 8:
        train_8gpu.remote(run_id=run_id, variant=variant)
    else:
        train_1gpu.remote(run_id=run_id, variant=variant)

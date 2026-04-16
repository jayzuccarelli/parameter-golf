import modal, os, sys, tempfile, subprocess
from pathlib import Path

app = modal.App("parameter-golf")
vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "sentencepiece", "huggingface_hub", "tqdm", "zstandard")
)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "sentencepiece", "huggingface_hub", "tqdm")
    .add_local_dir(str(Path(__file__).parent / "data"), remote_path="/app/data")
)

@app.function(image=data_image, volumes={"/data": vol}, timeout=7200, memory=4096)
def setup_data(train_shards: int = 10):
    import shutil
    os.makedirs("/data/datasets", exist_ok=True)
    os.makedirs("/data/tokenizers", exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "data"
    tmp.mkdir()
    shutil.copy("/app/data/cached_challenge_fineweb.py", tmp / "cached_challenge_fineweb.py")
    (tmp / "datasets").symlink_to("/data/datasets")
    (tmp / "tokenizers").symlink_to("/data/tokenizers")
    subprocess.run([sys.executable, str(tmp / "cached_challenge_fineweb.py"),
                    "--variant", "sp1024", "--train-shards", str(train_shards)], check=True)
    vol.commit()
    print("Done.")

@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=2400, memory=32768)
def run_experiment(code: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(code); tmp.flush()
    env = {**os.environ, "MAX_WALLCLOCK_SECONDS": "300",
           "DATA_PATH": "/data/datasets/fineweb10B_sp1024",
           "TOKENIZER_PATH": "/data/tokenizers/fineweb_1024_bpe.model",
           "ZSTD_LEVEL": "9",
           "EVAL_STRIDE": "0",
           "TRAIN_BATCH_TOKENS": "131072"}
    r = subprocess.run([sys.executable, tmp.name], capture_output=True, text=True, env=env, timeout=2400)
    return r.stdout + r.stderr

@app.local_entrypoint()
def run(script: str = "train_gpt.py"):
    """Run an experiment. Usage: modal run modal_runner.py --script train_gpt.py"""
    log = run_experiment.remote(Path(script).read_text())
    print(log)

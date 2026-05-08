"""
hf_auth.py — Shared HuggingFace authentication + model download utility.
Reads HF_TOKEN from .env in the same directory as this file.
"""

import os
import sys
from pathlib import Path

def load_env(env_path: str = None):
    """Parse a .env file and inject variables into os.environ."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        print(f"[WARN] .env file not found at {env_path}")
        print("       Create one with:  HF_TOKEN=hf_your_token_here")
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"\'\"')
            if key and value:
                os.environ.setdefault(key, value)

def login_huggingface():
    """
    Authenticate with HuggingFace using HF_TOKEN from the environment.
    Also triggers snapshot_download to cache the model locally on first run.
    """
    load_env()

    token = os.environ.get("HF_TOKEN", "")
    if not token or token == "hf_your_token_here":
        print("[ERROR] HF_TOKEN is not set or still has the placeholder value.")
        print("        Edit .env and set:  HF_TOKEN=hf_your_real_token")
        sys.exit(1)

    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[INFO] Logged in to HuggingFace successfully.")
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        sys.exit(1)

def ensure_model(model_id: str):
    """
    Check the model is accessible. huggingface_hub will download it to the
    local cache (~/.cache/huggingface/hub) on first use — subsequent loads
    are instant. Exits with a helpful message on access errors.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

    print(f"[INFO] Verifying model access: {model_id}")
    try:
        snapshot_download(
            repo_id        = model_id,
            repo_type      = "model",
            ignore_patterns= ["*.pt", "original/*"],  # skip redundant weight formats
        )
        print(f"[INFO] Model ready: {model_id}")
    except GatedRepoError:
        print(f"[ERROR] {model_id} is a gated model.")
        print("        Accept the license at https://huggingface.co/{model_id}")
        print("        then re-run.")
        sys.exit(1)
    except RepositoryNotFoundError:
        print(f"[ERROR] Model not found: {model_id}")
        print("        Check the model ID or your HF_TOKEN permissions.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        sys.exit(1)

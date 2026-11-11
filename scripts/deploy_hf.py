"""Puts the site online for free on Hugging Face Spaces (Docker), or updates it after a change.

    pip install huggingface_hub
    python -m scripts.deploy_hf            # Space name: dealhunter
    python -m scripts.deploy_hf my-name    # another name

Needs HF_TOKEN in .env: a free token with "Write" access from https://huggingface.co/settings/tokens
The API keys in .env are copied into the Space's secrets (Hugging Face keeps them hidden, they are never uploaded
as a file or printed). .env, data/, debug/ and tests/ are never uploaded.
"""
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SECRET_SUFFIXES = ("_API_KEY", "_ACCOUNT_ID")
NEVER_UPLOAD = [".env", ".env.*", ".git/*", ".github/*", ".venv/*", "data/*", "debug/*", "tests/*", "**/__pycache__/*",
                ".pytest_cache/*", "README.md"]
# The Space reads its settings from this README header
SPACE_README = """---
title: DealHunter
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

DealHunter finds the cheapest in-stock offers for any product in online stores.
Code: https://github.com/mahdii99/dealhunter
"""


def read_env() -> dict[str, str]:
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def main(name: str = "dealhunter") -> None:
    env = read_env()
    if not env.get("HF_TOKEN"):
        sys.exit("Add HF_TOKEN=... to .env first (a Write token from https://huggingface.co/settings/tokens)")
    api = HfApi(token=env["HF_TOKEN"])
    repo = f"{api.whoami()['name']}/{name}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)

    keys = [k for k in env if k.endswith(SECRET_SUFFIXES)]
    for k in keys:
        api.add_space_secret(repo, k, env[k])
    api.add_space_secret(repo, "CLIENT_SALT", env.get("CLIENT_SALT") or secrets.token_hex(16))
    api.add_space_variable(repo, "DAILY_LIMIT", env.get("DAILY_LIMIT", "5"))
    if env.get("MARKET"):
        api.add_space_variable(repo, "MARKET", env["MARKET"])
    print(f"Secrets set: {', '.join(keys) or 'none'} (+ CLIENT_SALT)")

    api.upload_folder(repo_id=repo, repo_type="space", folder_path=ROOT, ignore_patterns=NEVER_UPLOAD,
                      delete_patterns=["*.py", "*.html", "*.txt"], commit_message="Update DealHunter")
    api.upload_file(repo_id=repo, repo_type="space", path_or_fileobj=SPACE_README.encode(),
                    path_in_repo="README.md", commit_message="Space settings")
    print(f"Uploaded. It builds in a few minutes: https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main(*sys.argv[1:2])

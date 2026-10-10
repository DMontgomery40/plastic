"""Upload a local results folder to the private HF dataset repo (token from the HF_TOKEN env/secret).
python experiments/gpu/jobs/sync.py <local_dir> <path_in_repo>"""
import sys
from huggingface_hub import HfApi

REPO = "dmontgomery40/plastic-gpu-results"
try:
    HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=sys.argv[1], path_in_repo=sys.argv[2],
                          commit_message=f"sync {sys.argv[2]}")
    print("synced", sys.argv[2], flush=True)
except Exception as e:  # never kill the experiment because an upload failed
    print("sync failed:", type(e).__name__, str(e)[:200], flush=True)

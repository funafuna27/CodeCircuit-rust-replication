"""Phase 9, A3 step 0: fetch MBPP+ (evalplus/mbppplus on the Hugging Face Hub) into a new, empty
directory outside the repository, and record what was fetched.

The files are untrusted data. Nothing in them is executed: the test strings are parsed later by
p9_a3_mbppplus.py with ast.parse + ast.literal_eval only. This script only downloads and hashes.

  DEST=/home/user1/p9data/mbppplus   (must not exist, or be empty)
Writes DEST/FETCHED.json (repo id, revision, files, sha256) and copies it to
results/p9_state/a3_fetch.json.
"""
import hashlib
import json
import os
import shutil
import sys

DEST = os.environ.get('DEST', '/home/user1/p9data/mbppplus')
REPO = 'evalplus/mbppplus'
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(DEST) and os.listdir(DEST):
        sys.exit(f'{DEST} is not empty')
    os.makedirs(DEST, exist_ok=True)
    for k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(k, None)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().dataset_info(REPO)
    path = snapshot_download(repo_id=REPO, repo_type='dataset', revision=info.sha, local_dir=DEST)
    files = []
    for root, _, names in os.walk(path):
        for n in sorted(names):
            p = os.path.join(root, n)
            if '/.cache/' in p or n.startswith('.'):
                continue
            files.append({'path': os.path.relpath(p, DEST), 'bytes': os.path.getsize(p), 'sha256': sha256(p)})
    rec = {'repo': REPO, 'revision': info.sha, 'last_modified': str(info.last_modified), 'files': files}
    with open(os.path.join(DEST, 'FETCHED.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    os.makedirs(os.path.join(HERE, 'results', 'p9_state'), exist_ok=True)
    shutil.copy(os.path.join(DEST, 'FETCHED.json'), os.path.join(HERE, 'results', 'p9_state', 'a3_fetch.json'))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()

import json, os, glob, yaml

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def expand_glob(pattern):
    if os.path.isdir(pattern):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(pattern, ext)))
        return sorted(files)
    else:
        return sorted(glob.glob(pattern))

def write_jsonl(path: str, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r)+'\n')

def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    print(f"[DB] Appended record to {path}")

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


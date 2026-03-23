"""Cross-validate hf-fm inspect (remote) against Python HTTP Range requests.

For each test model, this script:
1. Discovers .safetensors files via the HF API
2. Fetches the header via two HTTP Range requests using Python urllib
3. Runs `hf-fm inspect <repo> <file> --json` (remote mode, no --cached)
4. Compares tensor count, names, dtypes, shapes, data_offsets, params, metadata

Usage: python tests/verify_inspect_remote.py
"""

import json
import os
import struct
import subprocess
import sys
import urllib.request
from functools import reduce
from operator import mul
from pathlib import Path


HF_FM = str(Path(__file__).resolve().parent.parent / "target" / "release" / "hf-fetch-model.exe")

# 5 small public models unlikely to be in the local cache
TEST_MODELS = [
    "openai-community/gpt2",
    "distilbert/distilgpt2",
    "albert/albert-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert/distilbert-base-uncased",
]


def hf_api_list_files(repo_id: str) -> list[dict]:
    """List files in a repo via the HF API."""
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "verify-inspect/1.0"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data.get("siblings", [])


def find_safetensors_file(repo_id: str) -> str | None:
    """Find the first .safetensors file in a repo (prefer model.safetensors)."""
    files = hf_api_list_files(repo_id)
    safetensors = [f["rfilename"] for f in files if f["rfilename"].endswith(".safetensors")]
    if not safetensors:
        return None
    # Prefer model.safetensors if it exists
    for name in safetensors:
        if name == "model.safetensors":
            return name
    return safetensors[0]


def fetch_header_python(repo_id: str, filename: str) -> dict:
    """Fetch safetensors header via two HTTP Range requests using Python."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"User-Agent": "verify-inspect/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Request 1: 8-byte length prefix
    headers1 = {**headers, "Range": "bytes=0-7"}
    req1 = urllib.request.Request(url, headers=headers1)
    with urllib.request.urlopen(req1) as resp1:
        len_bytes = resp1.read()
        # Extract file size from Content-Range header
        content_range = resp1.headers.get("Content-Range", "")
        file_size = None
        if "/" in content_range:
            try:
                file_size = int(content_range.split("/")[-1])
            except ValueError:
                pass

    if len(len_bytes) < 8:
        raise ValueError(f"Expected 8 bytes, got {len(len_bytes)}")

    header_size = struct.unpack("<Q", len_bytes[:8])[0]

    # Request 2: JSON header
    range_end = 8 + header_size - 1
    headers2 = {**headers, "Range": f"bytes=8-{range_end}"}
    req2 = urllib.request.Request(url, headers=headers2)
    with urllib.request.urlopen(req2) as resp2:
        json_bytes = resp2.read()

    header = json.loads(json_bytes)

    # Separate __metadata__
    metadata = header.pop("__metadata__", None)

    # Build tensor list sorted by offset
    tensors = []
    for name, info in header.items():
        tensors.append({
            "name": name,
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": tuple(info["data_offsets"]),
        })
    tensors.sort(key=lambda t: t["data_offsets"][0])

    return {
        "tensors": tensors,
        "metadata": metadata,
        "header_size": header_size,
        "file_size": file_size,
    }


def num_elements(shape: list) -> int:
    if not shape:
        return 1
    return reduce(mul, shape, 1)


def run_hf_fm_inspect_json(repo_id: str, filename: str) -> dict | None:
    """Run hf-fm inspect --json (remote, no --cached) and parse output."""
    try:
        result = subprocess.run(
            [HF_FM, "inspect", repo_id, filename, "--json"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"    hf-fm FAILED: {result.stderr.strip()}")
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"    hf-fm ERROR: {e}")
        return None


def compare(repo_id: str, filename: str) -> list[str]:
    """Compare Python Range requests vs hf-fm inspect. Returns errors."""
    errors = []

    # Python
    try:
        py = fetch_header_python(repo_id, filename)
    except Exception as e:
        return [f"Python Range request failed: {e}"]

    # Rust
    rust = run_hf_fm_inspect_json(repo_id, filename)
    if rust is None:
        return ["hf-fm inspect returned no output"]

    # header_size
    if py["header_size"] != rust["header_size"]:
        errors.append(f"header_size: Python={py['header_size']} vs Rust={rust['header_size']}")

    # file_size
    rust_fs = rust.get("file_size")
    if py["file_size"] is not None and rust_fs is not None and py["file_size"] != rust_fs:
        errors.append(f"file_size: Python={py['file_size']} vs Rust={rust_fs}")

    # tensor count
    if len(py["tensors"]) != len(rust["tensors"]):
        errors.append(f"tensor_count: Python={len(py['tensors'])} vs Rust={len(rust['tensors'])}")
        return errors

    # per-tensor comparison
    for i, (pt, rt) in enumerate(zip(py["tensors"], rust["tensors"])):
        if pt["name"] != rt["name"]:
            errors.append(f"tensor[{i}] name: '{pt['name']}' vs '{rt['name']}'")
        if pt["dtype"] != rt["dtype"]:
            errors.append(f"tensor[{i}] '{pt['name']}' dtype: '{pt['dtype']}' vs '{rt['dtype']}'")
        if pt["shape"] != rt["shape"]:
            errors.append(f"tensor[{i}] '{pt['name']}' shape: {pt['shape']} vs {rt['shape']}")
        if pt["data_offsets"] != tuple(rt["data_offsets"]):
            errors.append(f"tensor[{i}] '{pt['name']}' offsets: {pt['data_offsets']} vs {tuple(rt['data_offsets'])}")

    # total params
    py_params = sum(num_elements(t["shape"]) for t in py["tensors"])
    rust_params = sum(num_elements(t["shape"]) for t in rust["tensors"])
    if py_params != rust_params:
        errors.append(f"total_params: Python={py_params} vs Rust={rust_params}")

    # metadata
    py_meta = py["metadata"]
    rust_meta = rust.get("metadata")
    if (py_meta is None) != (rust_meta is None):
        errors.append(f"metadata presence: Python={'yes' if py_meta else 'no'} vs Rust={'yes' if rust_meta else 'no'}")
    elif py_meta is not None and rust_meta is not None:
        py_str = {k: str(v) for k, v in py_meta.items()}
        if py_str != rust_meta:
            errors.append(f"metadata mismatch: Python={py_str} vs Rust={rust_meta}")

    return errors


def main():
    print(f"hf-fm binary: {HF_FM}")
    print(f"Testing {len(TEST_MODELS)} remote models via HTTP Range requests")
    print()

    if not os.path.exists(HF_FM):
        print(f"ERROR: hf-fm binary not found at {HF_FM}")
        sys.exit(1)

    total_files = 0
    total_tensors = 0
    total_errors = 0
    failed = []

    for repo_id in TEST_MODELS:
        print(f"[{repo_id}]")

        # Find a .safetensors file
        try:
            filename = find_safetensors_file(repo_id)
        except Exception as e:
            print(f"  SKIP: cannot list files: {e}")
            continue

        if filename is None:
            print(f"  SKIP: no .safetensors files in repo")
            continue

        print(f"  File: {filename}")
        total_files += 1

        errors = compare(repo_id, filename)
        if errors:
            total_errors += len(errors)
            failed.append((repo_id, filename))
            for err in errors:
                print(f"  FAIL: {err}")
        else:
            py = fetch_header_python(repo_id, filename)
            n = len(py["tensors"])
            total_tensors += n
            params = sum(num_elements(t["shape"]) for t in py["tensors"])
            print(f"  OK:   {n} tensors, {params:,} params — Python and Rust match exactly")

        print()

    print("=" * 70)
    print(f"Models tested:    {total_files}")
    print(f"Tensors verified: {total_tensors:,}")
    print(f"Errors found:     {total_errors}")

    if failed:
        print()
        print("FAILED:")
        for repo, fname in failed:
            print(f"  {repo}/{fname}")
        sys.exit(1)
    else:
        print()
        print("ALL REMOTE CHECKS PASSED — Range request results match Python exactly.")
        sys.exit(0)


if __name__ == "__main__":
    main()

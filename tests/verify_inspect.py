"""Cross-validate hf-fm inspect output against Python safetensors header parsing.

For every cached .safetensors file, this script:
1. Reads the raw safetensors header (8-byte LE length prefix + JSON) using Python
2. Runs `hf-fm inspect <repo> <file> --cached --json` and parses the JSON output
3. Compares tensor count, tensor names, dtypes, shapes, data_offsets, param counts,
   header_size, file_size, and metadata — reports ANY discrepancy.

Usage: python tests/verify_inspect.py
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from functools import reduce
from operator import mul


HF_FM = str(Path(__file__).resolve().parent.parent / "target" / "release" / "hf-fetch-model.exe")
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def read_safetensors_header_python(filepath: Path) -> dict:
    """Read and parse safetensors header using pure Python (no dependencies)."""
    with open(filepath, "rb") as f:
        # 8-byte little-endian header length
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"File too short: {filepath}")
        header_size = struct.unpack("<Q", raw_len)[0]

        # Read JSON header
        json_bytes = f.read(header_size)
        if len(json_bytes) < header_size:
            raise ValueError(f"Truncated header in {filepath}")

    header = json.loads(json_bytes)

    # Separate __metadata__ from tensors
    metadata = header.pop("__metadata__", None)

    # Build tensor list sorted by data_offsets[0] (same order as Rust code)
    tensors = []
    for name, info in header.items():
        tensors.append({
            "name": name,
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": tuple(info["data_offsets"]),
        })
    tensors.sort(key=lambda t: t["data_offsets"][0])

    file_size = os.path.getsize(filepath)

    return {
        "tensors": tensors,
        "metadata": metadata,
        "header_size": header_size,
        "file_size": file_size,
    }


def num_elements(shape: list) -> int:
    """Product of shape dimensions."""
    if not shape:
        return 1
    return reduce(mul, shape, 1)


def run_hf_fm_inspect_json(repo_id: str, filename: str) -> dict | None:
    """Run hf-fm inspect --cached --json and parse the output."""
    try:
        result = subprocess.run(
            [HF_FM, "inspect", repo_id, filename, "--cached", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"    hf-fm FAILED: {result.stderr.strip()}")
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"    hf-fm ERROR: {e}")
        return None


def compare_headers(repo_id: str, filename: str, filepath: Path) -> list[str]:
    """Compare Python-parsed header vs hf-fm inspect output. Returns list of errors."""
    errors = []

    # 1. Parse with Python
    try:
        py = read_safetensors_header_python(filepath)
    except Exception as e:
        return [f"Python parse failed: {e}"]

    # 2. Parse with hf-fm
    rust = run_hf_fm_inspect_json(repo_id, filename)
    if rust is None:
        return ["hf-fm inspect returned no output"]

    # 3. Compare header_size
    if py["header_size"] != rust["header_size"]:
        errors.append(f"header_size: Python={py['header_size']} vs Rust={rust['header_size']}")

    # 4. Compare file_size
    rust_file_size = rust.get("file_size")
    if rust_file_size is not None and py["file_size"] != rust_file_size:
        errors.append(f"file_size: Python={py['file_size']} vs Rust={rust_file_size}")

    # 5. Compare tensor count
    py_count = len(py["tensors"])
    rust_count = len(rust["tensors"])
    if py_count != rust_count:
        errors.append(f"tensor_count: Python={py_count} vs Rust={rust_count}")
        return errors  # No point comparing individual tensors if counts differ

    # 6. Compare each tensor (name, dtype, shape, data_offsets)
    for i, (pt, rt) in enumerate(zip(py["tensors"], rust["tensors"])):
        if pt["name"] != rt["name"]:
            errors.append(f"tensor[{i}] name: Python='{pt['name']}' vs Rust='{rt['name']}'")
        if pt["dtype"] != rt["dtype"]:
            errors.append(f"tensor[{i}] '{pt['name']}' dtype: Python='{pt['dtype']}' vs Rust='{rt['dtype']}'")
        if pt["shape"] != rt["shape"]:
            errors.append(f"tensor[{i}] '{pt['name']}' shape: Python={pt['shape']} vs Rust={rt['shape']}")
        # data_offsets: Rust returns a tuple/list [start, end]
        rust_offsets = tuple(rt["data_offsets"])
        if pt["data_offsets"] != rust_offsets:
            errors.append(f"tensor[{i}] '{pt['name']}' data_offsets: Python={pt['data_offsets']} vs Rust={rust_offsets}")

    # 7. Compare total param count
    py_params = sum(num_elements(t["shape"]) for t in py["tensors"])
    rust_params = sum(num_elements(t["shape"]) for t in rust["tensors"])
    if py_params != rust_params:
        errors.append(f"total_params: Python={py_params} vs Rust={rust_params}")

    # 8. Compare metadata
    py_meta = py["metadata"]
    rust_meta = rust.get("metadata")
    if py_meta is None and rust_meta is not None:
        errors.append(f"metadata: Python=None vs Rust={rust_meta}")
    elif py_meta is not None and rust_meta is None:
        errors.append(f"metadata: Python={py_meta} vs Rust=None")
    elif py_meta is not None and rust_meta is not None:
        # Convert all Python metadata values to strings for comparison
        # (safetensors metadata values are always strings in the spec)
        py_meta_str = {k: str(v) for k, v in py_meta.items()}
        if py_meta_str != rust_meta:
            errors.append(f"metadata mismatch: Python={py_meta_str} vs Rust={rust_meta}")

    return errors


def find_all_cached_safetensors() -> list[tuple[str, str, Path]]:
    """Find all cached .safetensors files. Returns (repo_id, filename, filepath)."""
    results = []
    if not CACHE_DIR.exists():
        return results

    for model_dir in sorted(CACHE_DIR.iterdir()):
        dir_name = model_dir.name
        if not dir_name.startswith("models--"):
            continue

        repo_part = dir_name[len("models--"):]
        sep_pos = repo_part.find("--")
        if sep_pos == -1:
            continue
        repo_id = repo_part[:sep_pos] + "/" + repo_part[sep_pos + 2:]

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        for snap_dir in snapshots_dir.iterdir():
            if not snap_dir.is_dir():
                continue
            for sf in sorted(snap_dir.rglob("*.safetensors")):
                rel_name = str(sf.relative_to(snap_dir)).replace("\\", "/")
                results.append((repo_id, rel_name, sf))

    return results


def main():
    print(f"hf-fm binary: {HF_FM}")
    print(f"Cache dir:    {CACHE_DIR}")
    print()

    if not os.path.exists(HF_FM):
        print(f"ERROR: hf-fm binary not found at {HF_FM}")
        print("Run: cargo build --features cli --release")
        sys.exit(1)

    all_files = find_all_cached_safetensors()
    print(f"Found {len(all_files)} cached .safetensors files across {len(set(r for r,_,_ in all_files))} repos")
    print()

    total_files = 0
    total_errors = 0
    total_tensors_verified = 0
    failed_repos = []

    current_repo = None
    for repo_id, filename, filepath in all_files:
        if repo_id != current_repo:
            current_repo = repo_id
            print(f"[{repo_id}]")

        total_files += 1
        errors = compare_headers(repo_id, filename, filepath)

        if errors:
            total_errors += len(errors)
            failed_repos.append((repo_id, filename))
            for err in errors:
                print(f"  FAIL {filename}: {err}")
        else:
            py = read_safetensors_header_python(filepath)
            n = len(py["tensors"])
            total_tensors_verified += n
            total_params = sum(num_elements(t["shape"]) for t in py["tensors"])
            print(f"  OK   {filename}: {n} tensors, {total_params:,} params")

    print()
    print("=" * 70)
    print(f"Files verified:   {total_files}")
    print(f"Tensors verified: {total_tensors_verified:,}")
    print(f"Errors found:     {total_errors}")

    if failed_repos:
        print()
        print("FAILED FILES:")
        for repo, fname in failed_repos:
            print(f"  {repo}/{fname}")
        sys.exit(1)
    else:
        print()
        print("ALL CHECKS PASSED — hf-fm inspect output matches Python exactly.")
        sys.exit(0)


if __name__ == "__main__":
    main()

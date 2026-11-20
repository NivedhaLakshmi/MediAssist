"""Hardware footprint of each local model.

For every model: size on disk, memory used once loaded (at the app's
8192-token context), time to load from cold, and generation speed on a
fixed short prompt. Each model is unloaded first so load time is cold
start from disk (the OS file cache may still help on repeat runs).

Run: venv/bin/python experiments/eval/eval_footprint.py
"""

import json
import platform
import subprocess
import time

import requests

from common import save

API = "http://127.0.0.1:11434"
MODELS = ["llama3.2:latest", "llama3:latest", "qwen2.5:7b-instruct", "medgemma:4b"]
PROMPT = "In one sentence, what does a high white blood cell count usually suggest?"
RUNS = 3


def unload_all():
    for m in requests.get(f"{API}/api/ps", timeout=10).json().get("models", []):
        requests.post(f"{API}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    time.sleep(2)


def disk_size(model):
    for m in requests.get(f"{API}/api/tags", timeout=10).json()["models"]:
        if m["name"] == model:
            return m["size"]
    return None


def loaded(model):
    for m in requests.get(f"{API}/api/ps", timeout=10).json().get("models", []):
        if m["name"] == model:
            return m
    return None


def main():
    results = {}
    for model in MODELS:
        loads, speeds = [], []
        mem = vram = None
        for _ in range(RUNS):
            unload_all()
            start = time.monotonic()
            r = requests.post(f"{API}/api/generate", json={
                "model": model, "prompt": PROMPT, "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 60},
                "keep_alive": "5m"}, timeout=600).json()
            wall = time.monotonic() - start
            loads.append(r.get("load_duration", 0) / 1e9)
            if r.get("eval_duration"):
                speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))
            info = loaded(model)
            if info:
                mem, vram = info.get("size"), info.get("size_vram")
            print(f"{model}: load {loads[-1]:.2f}s, {speeds[-1] if speeds else 0:.1f} tok/s, wall {wall:.1f}s")
        results[model] = {
            "disk_gb": round(disk_size(model) / 1e9, 2),
            "memory_gb": round(mem / 1e9, 2) if mem else None,
            "on_gpu_share": round(vram / mem, 2) if mem and vram is not None else None,
            "load_seconds_median": round(sorted(loads)[len(loads) // 2], 2),
            "tokens_per_second_median": round(sorted(speeds)[len(speeds) // 2], 1) if speeds else None,
            "runs": RUNS,
        }
    unload_all()
    hw = {"machine": platform.machine(),
          "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
          "ram_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30)}
    out = {"hardware": hw, "context_tokens": 8192, "models": results}
    save("footprint", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

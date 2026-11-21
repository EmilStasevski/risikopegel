"""Minimal client for a local Ollama server (no data leaves the machine) and a GPU power sampler."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

URL = "http://localhost:11434"


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


def chat(model: str, messages: list[dict[str, str]], *, seed: int, temperature: float = 0.7) -> Reply:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"seed": seed, "temperature": temperature, "num_predict": 400},
    }
    req = urllib.request.Request(
        f"{URL}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out: dict[str, Any] = json.loads(r.read())
    return Reply(
        text=str(out["message"]["content"]).strip(),
        prompt_tokens=int(out.get("prompt_eval_count", 0)),
        output_tokens=int(out.get("eval_count", 0)),
        seconds=time.perf_counter() - t0,
    )


def model_digest(model: str) -> str:
    with urllib.request.urlopen(f"{URL}/api/tags", timeout=30) as r:
        tags = json.loads(r.read())["models"]
    return next((str(m["digest"]) for m in tags if m["name"] == model), "unknown")


class GpuPower:
    """Samples board power with nvidia-smi every 100 ms; joules() integrates the samples."""

    def __enter__(self) -> GpuPower:
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.t0
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        vals = [float(x) for x in out.split() if x.replace(".", "", 1).isdigit()]
        self.mean_watts = sum(vals) / len(vals) if vals else float("nan")

    @property
    def joules(self) -> float:
        return self.mean_watts * self.seconds

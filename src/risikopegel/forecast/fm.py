"""Zero-shot time-series foundation models (CPU), loaded from the local Hugging Face cache at pinned
revisions (downloaded once, approved 2026-11-21; see docs/data.md). No fine-tuning, no network at run time."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]

FM_MODELS: dict[str, tuple[str, str]] = {
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "chronos_bolt_small": ("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8"),
    "timesfm25": ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"),
}
QUANTILES = (0.1, 0.5, 0.9)


def local_path(name: str) -> str:
    from huggingface_hub import snapshot_download

    repo, rev = FM_MODELS[name]
    return str(snapshot_download(repo, revision=rev, local_files_only=True))


def load(name: str, h: int, max_context: int = 512) -> Callable[[Sequence[Arr]], Arr]:
    """Returns predict(histories) -> array (n_series, h, 3) of the 0.1/0.5/0.9 quantiles."""
    import torch

    torch.manual_seed(0)
    path = local_path(name)
    if name.startswith("chronos"):
        from chronos import BaseChronosPipeline

        pipe: Any = BaseChronosPipeline.from_pretrained(path, device_map="cpu")

        def predict_chronos(hist: Sequence[Arr]) -> Arr:
            ctx = [torch.tensor(x[-max_context:], dtype=torch.float32) for x in hist]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=list(QUANTILES))
            arr = [np.asarray(t, dtype=np.float64).reshape(h, 3) for t in q]
            return np.sort(np.stack(arr), axis=2)

        return predict_chronos

    import timesfm

    model: Any = timesfm.TimesFM_2p5_200M_torch.from_pretrained(path)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=128,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )

    def predict_timesfm(hist: Sequence[Arr]) -> Arr:
        _, qs = model.forecast(horizon=h, inputs=[x[-max_context:] for x in hist])
        qs = np.asarray(qs, dtype=np.float64)  # (n, h, 10): mean, then deciles 0.1 .. 0.9
        return np.sort(qs[:, :, [1, 5, 9]], axis=2)

    return predict_timesfm

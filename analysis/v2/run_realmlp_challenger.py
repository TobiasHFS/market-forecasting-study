"""Strict chronological RealMLP-style challenger for the v2 model search.

The implementation follows the small official RealMLP-TD-S recipe (learnable
input scales, neural-tangent-parameterized linear layers, Mish, target
standardization, Adam parameter groups) but adapts batch size and epoch count
to this 1.26M-row dataset.  Outer validation months are never used for epoch
selection: the last ``inner_val_months`` of each outer training window form an
inner chronological validation set.

All outputs are isolated under ``artifacts/v2/realmlp``.  The accepted v1
pipeline and submission are read-only inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = WORKSPACE_ROOT / "analysis"
LOCAL_DEPS = WORKSPACE_ROOT / ".analysis_deps"
for dependency_path in (LOCAL_DEPS, ANALYSIS_ROOT):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from feature_families import materialize_feature_set
from modeling import (
    DEVELOPMENT_FOLDS,
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    rms_normalize,
    summarize_monthly_diagnostics,
)
from pipeline_config import DATA_ROOT, RANDOM_SEED


OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "v2" / "realmlp"
BASELINE_OOF = (
    WORKSPACE_ROOT
    / "artifacts"
    / "diagnostics"
    / "gbdt_multiscale_mechanics_scale_development_oof.npz"
)


@dataclass(frozen=True)
class NeuralSpec:
    feature_set: str = "multiscale_mechanics_scale"
    hidden_size: int = 256
    hidden_layers: int = 3
    batch_size: int = 2048
    inference_batch_size: int = 8192
    max_epochs: int = 64
    min_epochs: int = 12
    patience: int = 10
    base_lr: float = 0.04
    inner_val_months: int = 3
    smooth_clip_scale: float = 3.0
    prediction_power: float = 1.2
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.batch_size <= 0 or self.inference_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_epochs <= 0 or self.min_epochs <= 0:
            raise ValueError("epoch counts must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")
        if self.patience <= 0 or self.base_lr <= 0.0:
            raise ValueError("patience and base_lr must be positive")
        if self.inner_val_months <= 0:
            raise ValueError("inner_val_months must be positive")
        if not np.isfinite(self.smooth_clip_scale) or self.smooth_clip_scale <= 0.0:
            raise ValueError("smooth_clip_scale must be finite and positive")
        if not np.isfinite(self.prediction_power) or self.prediction_power <= 0.0:
            raise ValueError("prediction_power must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    sample_id = table["sample_id"].to_numpy(zero_copy_only=False)
    months = table["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = table["target"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    expected = np.arange(len(sample_id), dtype=sample_id.dtype)
    if not np.array_equal(sample_id, expected):
        raise ValueError("label sample_id is not exact row alignment")
    if np.any(months[1:] < months[:-1]) or not np.all(np.isfinite(target)):
        raise ValueError("invalid label chronology or target")
    return months, target


def _month_slice(months: np.ndarray, start: int, end: int) -> slice:
    left = int(np.searchsorted(months, start, side="left"))
    right = int(np.searchsorted(months, end, side="right"))
    if left >= right or int(months[left]) != start or int(months[right - 1]) != end:
        raise ValueError(f"incomplete month slice {start}..{end}")
    return slice(left, right)


def _power_view(prediction: np.ndarray, power: float) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    return np.sign(values) * np.power(np.abs(values), power)


def _smooth_clip_inplace(matrix: np.ndarray, scale: float) -> None:
    """Apply RealMLP's smooth saturation in bounded row chunks."""

    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise ValueError("smooth clipping expects a two-dimensional float32 matrix")
    for left in range(0, matrix.shape[0], 65_536):
        right = min(left + 65_536, matrix.shape[0])
        block = matrix[left:right]
        denominator = np.sqrt(1.0 + np.square(block / scale, dtype=np.float32))
        np.divide(block, denominator, out=block)


def _set_determinism(torch: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _build_model(torch: Any, input_dim: int, spec: NeuralSpec) -> Any:
    nn = torch.nn

    class ScalingLayer(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_features))

        def forward(self, x: Any) -> Any:
            return x * self.scale.unsqueeze(0)

    class NTPLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, zero_init: bool = False) -> None:
            super().__init__()
            factor = 0.0 if zero_init else 1.0
            self.in_features = in_features
            self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
            self.bias = nn.Parameter(factor * torch.randn(1, out_features))

        def forward(self, x: Any) -> Any:
            return (x @ self.weight) / math.sqrt(self.in_features) + self.bias

    class Mish(nn.Module):
        def forward(self, x: Any) -> Any:
            return x * torch.tanh(torch.nn.functional.softplus(x))

    layers: list[Any] = [ScalingLayer(input_dim)]
    in_features = input_dim
    for _ in range(spec.hidden_layers):
        layers.extend([NTPLinear(in_features, spec.hidden_size), Mish()])
        in_features = spec.hidden_size
    layers.append(NTPLinear(in_features, 1, zero_init=True))
    return nn.Sequential(*layers)


def _predict_batches(
    torch: Any,
    model: Any,
    matrix: np.ndarray,
    *,
    device: Any,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    result = np.empty(matrix.shape[0], dtype=np.float64)
    with torch.no_grad():
        for left in range(0, matrix.shape[0], batch_size):
            right = min(left + batch_size, matrix.shape[0])
            batch = torch.as_tensor(matrix[left:right], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(batch).squeeze(-1)
            result[left:right] = prediction.float().cpu().numpy()
    return result


def _make_optimizer(torch: Any, model: Any) -> Any:
    parameters = list(model.parameters())
    return torch.optim.Adam(
        [
            {"params": [parameters[0]]},
            {"params": parameters[1::2]},
            {"params": parameters[2::2]},
        ],
        betas=(0.9, 0.95),
    )


def _run_training_epoch(
    torch: Any,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    criterion: Any,
    X_tensor: Any,
    y_tensor: Any,
    data_device: Any,
    device: Any,
    spec: NeuralSpec,
    epoch: int,
    schedule_horizon_epochs: int,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one RealMLP epoch and return mean loss and final learning rate."""

    model.train()
    n_train = int(X_tensor.shape[0])
    n_batches = math.ceil(n_train / spec.batch_size)
    total_steps = schedule_horizon_epochs * n_batches
    permutation = torch.randperm(n_train, device=data_device)
    weighted_loss = 0.0
    observed = 0
    learning_rate = 0.0
    for batch_index, left in enumerate(range(0, n_train, spec.batch_size)):
        indices = permutation[left : left + spec.batch_size]
        x_batch = X_tensor[indices]
        y_batch = y_tensor[indices]
        if data_device.type != device.type:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
        global_step = (epoch - 1) * n_batches + batch_index
        progress = global_step / max(total_steps - 1, 1)
        schedule = 0.5 - 0.5 * math.cos(
            2.0 * math.pi * math.log2(1.0 + 15.0 * progress)
        )
        learning_rate = spec.base_lr * schedule
        optimizer.param_groups[0]["lr"] = 6.0 * learning_rate
        optimizer.param_groups[1]["lr"] = learning_rate
        optimizer.param_groups[2]["lr"] = 0.1 * learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            standardized_prediction = model(x_batch).squeeze(-1)
            loss = criterion(standardized_prediction, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_rows = int(y_batch.shape[0])
        weighted_loss += float(loss.detach().cpu()) * batch_rows
        observed += batch_rows
    return weighted_loss / observed, learning_rate


def _train_one_fold(
    torch: Any,
    *,
    X_raw: np.ndarray,
    names: Sequence[str],
    kinds: Sequence[str],
    months: np.ndarray,
    target: np.ndarray,
    fold: Any,
    spec: NeuralSpec,
    device: Any,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    fold_prediction_path = OUTPUT_ROOT / f"{fold.name.lower()}_predictions.npz"
    fold_summary_path = OUTPUT_ROOT / f"{fold.name.lower()}_summary.json"
    if fold_prediction_path.exists() and fold_summary_path.exists() and not overwrite:
        cached = np.load(fold_prediction_path, allow_pickle=False)
        prediction = cached["prediction"].astype(np.float64, copy=False)
        summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
        return prediction, summary, summary["epochs"] + summary.get("refit_epochs", [])

    inner_validation_start = fold.train_end - spec.inner_val_months + 1
    if inner_validation_start <= fold.train_start:
        raise ValueError(f"{fold.name}: inner validation consumes the training window")
    fit_slice = _month_slice(months, fold.train_start, inner_validation_start - 1)
    inner_slice = _month_slice(months, inner_validation_start, fold.train_end)
    outer_train_slice = _month_slice(months, fold.train_start, fold.train_end)
    outer_slice = _month_slice(months, fold.validation_start, fold.validation_end)

    transform_started = time.perf_counter()
    preprocessor = RobustPreprocessor(
        feature_names=list(names),
        feature_kinds=list(kinds),
        clip=8.0,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )
    preprocessor.fit(X_raw[fit_slice])
    X_fit = np.ascontiguousarray(preprocessor.transform(X_raw[fit_slice]))
    X_inner = np.ascontiguousarray(preprocessor.transform(X_raw[inner_slice]))
    _smooth_clip_inplace(X_fit, spec.smooth_clip_scale)
    _smooth_clip_inplace(X_inner, spec.smooth_clip_scale)
    selection_transform_seconds = time.perf_counter() - transform_started

    y_fit = target[fit_slice]
    y_inner = target[inner_slice]
    y_outer = target[outer_slice]
    target_mean = float(np.mean(y_fit))
    target_std = float(np.std(y_fit))
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError(f"{fold.name}: invalid target standard deviation")
    y_fit_standard = ((y_fit - target_mean) / target_std).astype(np.float32)

    fold_seed = spec.seed + int(fold.name[-1])
    _set_determinism(torch, fold_seed)
    model = _build_model(torch, X_fit.shape[1], spec).to(device)
    optimizer = _make_optimizer(torch, model)
    criterion = torch.nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        available, _total = torch.cuda.mem_get_info(device)
        required = X_fit.nbytes + y_fit_standard.nbytes
        cache_on_device = required < int(available * 0.70)
    else:
        cache_on_device = True
    data_device = device if cache_on_device else torch.device("cpu")
    X_fit_tensor = torch.as_tensor(X_fit, dtype=torch.float32, device=data_device)
    y_fit_tensor = torch.as_tensor(y_fit_standard, dtype=torch.float32, device=data_device)
    if cache_on_device:
        del X_fit, y_fit_standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_inner_cosine = -np.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    selection_training_started = time.perf_counter()

    for epoch in range(1, spec.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, learning_rate = _run_training_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            X_tensor=X_fit_tensor,
            y_tensor=y_fit_tensor,
            data_device=data_device,
            device=device,
            spec=spec,
            epoch=epoch,
            schedule_horizon_epochs=spec.max_epochs,
            use_amp=use_amp,
        )

        inner_standardized = _predict_batches(
            torch,
            model,
            X_inner,
            device=device,
            batch_size=spec.inference_batch_size,
            use_amp=use_amp,
        )
        inner_prediction = inner_standardized * target_std + target_mean
        inner_cosine = cosine_score(y_inner, inner_prediction)
        inner_mse = float(np.mean(np.square(y_inner - inner_prediction)))
        improved = inner_cosine > best_inner_cosine + 1e-7
        if improved:
            best_inner_cosine = inner_cosine
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "fold": fold.name,
            "stage": "selection",
            "epoch": epoch,
            "train_mse_standardized": train_loss,
            "inner_cosine": inner_cosine,
            "inner_mse": inner_mse,
            "learning_rate_last": learning_rate,
            "improved": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_rows.append(row)
        print(
            f"{fold.name} epoch {epoch:03d}: train_mse={row['train_mse_standardized']:.6f} "
            f"inner_cos={inner_cosine:.6f} best={best_inner_cosine:.6f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if epoch >= spec.min_epochs and epochs_without_improvement >= spec.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{fold.name}: no finite best checkpoint")

    selection_training_seconds = time.perf_counter() - selection_training_started
    transformed_feature_count = int(X_fit_tensor.shape[1])
    del model, optimizer, scaler, X_fit_tensor, y_fit_tensor, X_inner, preprocessor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Honest refit: the inner window selected only the epoch count. Start from
    # scratch, refit preprocessing on every outer-training month, and train for
    # exactly that count before touching outer-validation labels.
    refit_transform_started = time.perf_counter()
    refit_preprocessor = RobustPreprocessor(
        feature_names=list(names),
        feature_kinds=list(kinds),
        clip=8.0,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )
    refit_preprocessor.fit(X_raw[outer_train_slice])
    X_refit = np.ascontiguousarray(
        refit_preprocessor.transform(X_raw[outer_train_slice])
    )
    X_outer = np.ascontiguousarray(refit_preprocessor.transform(X_raw[outer_slice]))
    _smooth_clip_inplace(X_refit, spec.smooth_clip_scale)
    _smooth_clip_inplace(X_outer, spec.smooth_clip_scale)
    refit_transform_seconds = time.perf_counter() - refit_transform_started

    y_refit = target[outer_train_slice]
    refit_target_mean = float(np.mean(y_refit))
    refit_target_std = float(np.std(y_refit))
    if not np.isfinite(refit_target_std) or refit_target_std <= 0.0:
        raise ValueError(f"{fold.name}: invalid refit target standard deviation")
    y_refit_standard = ((y_refit - refit_target_mean) / refit_target_std).astype(
        np.float32
    )
    refit_transformed_feature_count = int(X_refit.shape[1])

    _set_determinism(torch, fold_seed)
    refit_model = _build_model(torch, X_refit.shape[1], spec).to(device)
    refit_optimizer = _make_optimizer(torch, refit_model)
    refit_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        available, _total = torch.cuda.mem_get_info(device)
        required = X_refit.nbytes + y_refit_standard.nbytes
        refit_cache_on_device = required < int(available * 0.70)
    else:
        refit_cache_on_device = True
    refit_data_device = device if refit_cache_on_device else torch.device("cpu")
    X_refit_tensor = torch.as_tensor(
        X_refit, dtype=torch.float32, device=refit_data_device
    )
    y_refit_tensor = torch.as_tensor(
        y_refit_standard, dtype=torch.float32, device=refit_data_device
    )
    if refit_cache_on_device:
        del X_refit, y_refit_standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    refit_epoch_rows: list[dict[str, Any]] = []
    refit_training_started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        epoch_started = time.perf_counter()
        train_loss, learning_rate = _run_training_epoch(
            torch,
            model=refit_model,
            optimizer=refit_optimizer,
            scaler=refit_scaler,
            criterion=criterion,
            X_tensor=X_refit_tensor,
            y_tensor=y_refit_tensor,
            data_device=refit_data_device,
            device=device,
            spec=spec,
            epoch=epoch,
            schedule_horizon_epochs=spec.max_epochs,
            use_amp=use_amp,
        )
        refit_row = {
            "fold": fold.name,
            "stage": "refit",
            "epoch": epoch,
            "train_mse_standardized": train_loss,
            "learning_rate_last": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        refit_epoch_rows.append(refit_row)
        print(
            f"{fold.name} refit epoch {epoch:03d}/{best_epoch:03d}: "
            f"train_mse={train_loss:.6f} seconds={refit_row['epoch_seconds']:.1f}",
            flush=True,
        )
    refit_training_seconds = time.perf_counter() - refit_training_started

    outer_standardized = _predict_batches(
        torch,
        refit_model,
        X_outer,
        device=device,
        batch_size=spec.inference_batch_size,
        use_amp=use_amp,
    )
    outer_prediction = outer_standardized * refit_target_std + refit_target_mean
    if not np.all(np.isfinite(outer_prediction)):
        raise ValueError(f"{fold.name}: non-finite outer predictions")

    summary: dict[str, Any] = {
        "fold": fold.name,
        "outer_train_months": [fold.train_start, fold.train_end],
        "inner_fit_months": [fold.train_start, inner_validation_start - 1],
        "inner_validation_months": [inner_validation_start, fold.train_end],
        "outer_validation_months": [fold.validation_start, fold.validation_end],
        "selection_fit_rows": int(fit_slice.stop - fit_slice.start),
        "inner_validation_rows": int(inner_slice.stop - inner_slice.start),
        "refit_rows": int(outer_train_slice.stop - outer_train_slice.start),
        "outer_validation_rows": int(outer_slice.stop - outer_slice.start),
        "raw_features": int(X_raw.shape[1]),
        "selection_transformed_features": transformed_feature_count,
        "refit_transformed_features": refit_transformed_feature_count,
        "selection_target_mean": target_mean,
        "selection_target_std": target_std,
        "refit_target_mean": refit_target_mean,
        "refit_target_std": refit_target_std,
        "device": str(device),
        "selection_cache_training_matrix_on_device": cache_on_device,
        "refit_cache_training_matrix_on_device": refit_cache_on_device,
        "amp": use_amp,
        "best_epoch": best_epoch,
        "best_inner_cosine": best_inner_cosine,
        "refit_from_scratch_on_all_outer_train_months": True,
        "outer_cosine_raw": cosine_score(y_outer, outer_prediction),
        "outer_cosine_power": cosine_score(
            y_outer, _power_view(outer_prediction, spec.prediction_power)
        ),
        "selection_transform_seconds": selection_transform_seconds,
        "selection_training_seconds": selection_training_seconds,
        "refit_transform_seconds": refit_transform_seconds,
        "refit_training_seconds": refit_training_seconds,
        "epochs": epoch_rows,
        "refit_epochs": refit_epoch_rows,
    }
    np.savez_compressed(
        fold_prediction_path,
        row_indices=np.arange(outer_slice.start, outer_slice.stop, dtype=np.int64),
        months=months[outer_slice],
        target=y_outer,
        prediction=outer_prediction,
    )
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in refit_model.state_dict().items()
            },
            "input_dim": refit_transformed_feature_count,
            "target_mean": refit_target_mean,
            "target_std": refit_target_std,
            "spec": asdict(spec),
            "fold": fold.name,
            "selected_epoch": best_epoch,
            "refit_from_scratch_on_all_outer_train_months": True,
        },
        OUTPUT_ROOT / f"{fold.name.lower()}_checkpoint.pt",
    )
    fold_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del (
        refit_model,
        refit_optimizer,
        refit_scaler,
        X_refit_tensor,
        y_refit_tensor,
        X_outer,
        refit_preprocessor,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outer_prediction, summary, epoch_rows + refit_epoch_rows


def _candidate_comparison(
    target: np.ndarray,
    months: np.ndarray,
    realmlp_prediction: np.ndarray,
    row_indices: np.ndarray,
    power_grid: Sequence[float],
    blend_grid: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    baseline = np.load(BASELINE_OOF, allow_pickle=False)
    if not np.array_equal(row_indices, baseline["row_indices"]):
        raise ValueError("RealMLP OOF rows do not align with baseline OOF")
    if not np.array_equal(months, baseline["months"]):
        raise ValueError("RealMLP OOF months do not align with baseline OOF")
    if not np.array_equal(target, baseline["target"]):
        raise ValueError("RealMLP OOF targets do not exactly align with baseline OOF")

    views: dict[str, np.ndarray] = {
        "lightgbm_slow": baseline["slow"].astype(np.float64),
        "realmlp_raw": realmlp_prediction.astype(np.float64),
    }
    for power in power_grid:
        views[f"realmlp_power_{power:g}"] = _power_view(realmlp_prediction, power)

    selection_mask = months <= DEVELOPMENT_FOLDS[1].validation_end
    audit_mask = months >= DEVELOPMENT_FOLDS[2].validation_start
    best_realmlp_name = max(
        (name for name in views if name.startswith("realmlp")),
        key=lambda name: cosine_score(target[selection_mask], views[name][selection_mask]),
    )
    normalized_gbdt = rms_normalize(views["lightgbm_slow"])
    normalized_realmlp = rms_normalize(views[best_realmlp_name])
    for realmlp_weight in blend_grid:
        name = f"blend_realmlp_{realmlp_weight:g}"
        views[name] = rms_normalize(
            (1.0 - realmlp_weight) * normalized_gbdt
            + realmlp_weight * normalized_realmlp
        )

    rows: list[dict[str, Any]] = []
    for name, prediction in views.items():
        diagnostics = monthly_diagnostics(target, prediction, months)
        summary = summarize_monthly_diagnostics(
            diagnostics, cosine_score(target, prediction)
        ).to_dict()
        summary["candidate"] = name
        summary["selection_cosine_dev1_dev2"] = cosine_score(
            target[selection_mask], prediction[selection_mask]
        )
        summary["audit_cosine_dev3"] = cosine_score(
            target[audit_mask], prediction[audit_mask]
        )
        for fold in DEVELOPMENT_FOLDS:
            mask = (months >= fold.validation_start) & (months <= fold.validation_end)
            summary[f"{fold.name.lower()}_cosine"] = cosine_score(
                target[mask], prediction[mask]
            )
        rows.append(summary)
    ranking = pd.DataFrame(rows).sort_values(
        "selection_cosine_dev1_dev2", ascending=False, ignore_index=True
    )
    return ranking, views


def evaluate(spec: NeuralSpec, fold_names: Sequence[str], overwrite: bool) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is unavailable. Install the v2 runtime before running this challenger."
        ) from exc

    _set_determinism(torch, spec.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_folds = [fold for fold in DEVELOPMENT_FOLDS if fold.name in set(fold_names)]
    if len(selected_folds) != len(set(fold_names)):
        known = {fold.name for fold in DEVELOPMENT_FOLDS}
        raise ValueError(f"unknown or duplicate folds; known folds are {sorted(known)}")

    months, target = _load_labels()
    materialized = materialize_feature_set("train", spec.feature_set)
    X_raw = materialized.matrix
    fold_predictions: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    row_indices: list[np.ndarray] = []
    started = time.perf_counter()
    for fold in selected_folds:
        prediction, summary, epochs = _train_one_fold(
            torch,
            X_raw=X_raw,
            names=materialized.names,
            kinds=materialized.kinds,
            months=months,
            target=target,
            fold=fold,
            spec=spec,
            device=device,
            overwrite=overwrite,
        )
        fold_predictions.append(prediction)
        fold_summaries.append(summary)
        epoch_rows.extend(epochs)
        validation_slice = _month_slice(months, fold.validation_start, fold.validation_end)
        row_indices.append(
            np.arange(validation_slice.start, validation_slice.stop, dtype=np.int64)
        )

    oof_rows = np.concatenate(row_indices)
    oof_prediction = np.concatenate(fold_predictions)
    oof_months = months[oof_rows]
    oof_target = target[oof_rows]
    np.savez_compressed(
        OUTPUT_ROOT / "realmlp_development_oof.npz",
        row_indices=oof_rows,
        months=oof_months,
        target=oof_target,
        prediction=oof_prediction,
    )
    pd.DataFrame(epoch_rows).to_csv(OUTPUT_ROOT / "training_history.csv", index=False)

    complete_development_oof = (
        len(selected_folds) == len(DEVELOPMENT_FOLDS)
        and np.array_equal(
            oof_rows,
            np.load(BASELINE_OOF, allow_pickle=False)["row_indices"],
        )
    )
    ranking: pd.DataFrame | None = None
    if complete_development_oof:
        ranking, views = _candidate_comparison(
            oof_target,
            oof_months,
            oof_prediction,
            oof_rows,
            power_grid=(1.0, 1.1, 1.2, 1.3),
            blend_grid=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0),
        )
        ranking.to_csv(OUTPUT_ROOT / "candidate_comparison.csv", index=False)
        best_name = str(ranking.iloc[0]["candidate"])
        np.savez_compressed(
            OUTPUT_ROOT / "candidate_views.npz",
            row_indices=oof_rows,
            months=oof_months,
            target=oof_target,
            **views,
        )
    else:
        best_name = "not_available_partial_fold_run"

    payload: dict[str, Any] = {
        "experiment": "realmlp_style_strict_chronological_challenger",
        "spec": asdict(spec),
        "selected_folds": [fold.name for fold in selected_folds],
        "complete_development_oof": complete_development_oof,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "raw_feature_count": int(X_raw.shape[1]),
        "folds": fold_summaries,
        "partial_or_pooled_realmlp_cosine": cosine_score(oof_target, oof_prediction),
        "partial_or_pooled_power_cosine": cosine_score(
            oof_target, _power_view(oof_prediction, spec.prediction_power)
        ),
        "best_candidate": best_name,
        "ranking": None if ranking is None else ranking.to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - started,
        "source_references": [
            "https://github.com/dholzmueller/realmlp-td-s_standalone",
            "https://arxiv.org/abs/2407.04491",
        ],
    }
    summary_path = OUTPUT_ROOT / "realmlp_challenger_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_sha256"] = _sha256(summary_path)
    print(
        f"RealMLP challenger ({','.join(fold_names)}): cosine="
        f"{payload['partial_or_pooled_realmlp_cosine']:.6f}, device={device}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    defaults = NeuralSpec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=[fold.name for fold in DEVELOPMENT_FOLDS],
        default=[fold.name for fold in DEVELOPMENT_FOLDS],
    )
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--min-epochs", type=int, default=defaults.min_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--base-lr", type=float, default=defaults.base_lr)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = NeuralSpec(
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        base_lr=args.base_lr,
    )
    evaluate(spec, args.folds, args.overwrite)


if __name__ == "__main__":
    main()

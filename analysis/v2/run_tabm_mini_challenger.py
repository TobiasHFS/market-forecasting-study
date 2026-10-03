"""TabM-mini challenger with strict chronological selection and honest refit.

The architecture follows the official ICLR 2025 MiniEnsemble construction:
member-specific input affine scales, one shared MLP backbone, and independent
member heads.  Its loss is the mean of the k individual-member MSE values; the
member predictions are averaged only for validation scoring and inference.

All outputs are isolated under ``artifacts/v2/tabm_mini``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = WORKSPACE_ROOT / "analysis"
LOCAL_DEPS = WORKSPACE_ROOT / ".analysis_deps"
V2_ROOT = ANALYSIS_ROOT / "v2"
for dependency_path in (LOCAL_DEPS, ANALYSIS_ROOT, V2_ROOT):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd

from feature_families import materialize_feature_set
from modeling import (
    DEVELOPMENT_FOLDS,
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    rms_normalize,
    summarize_monthly_diagnostics,
)
from pipeline_config import RANDOM_SEED
from run_realmlp_challenger import (
    BASELINE_OOF,
    _load_labels,
    _month_slice,
    _power_view,
    _set_determinism,
    _sha256,
    _smooth_clip_inplace,
)


OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "v2" / "tabm_mini"


@dataclass(frozen=True)
class TabMMiniSpec:
    feature_set: str = "multiscale_mechanics_scale"
    k: int = 16
    hidden_size: int = 256
    hidden_layers: int = 2
    dropout: float = 0.10
    batch_size: int = 1024
    inference_batch_size: int = 4096
    max_epochs: int = 48
    min_epochs: int = 8
    patience: int = 8
    learning_rate: float = 0.002
    weight_decay: float = 0.0003
    inner_val_months: int = 3
    smooth_clip_scale: float = 3.0
    prediction_power: float = 1.2
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.k <= 1:
            raise ValueError("k must exceed one")
        if self.hidden_size <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size <= 0 or self.inference_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_epochs <= 0 or self.min_epochs <= 0:
            raise ValueError("epoch counts must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer parameters")
        if self.inner_val_months <= 0:
            raise ValueError("inner_val_months must be positive")


def _build_tabm_mini(torch: Any, input_dim: int, spec: TabMMiniSpec) -> Any:
    nn = torch.nn

    class MiniEnsemble(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Official MiniEnsemble order: diversify before features are mixed.
            self.input_scale = nn.Parameter(torch.empty(spec.k, input_dim))
            nn.init.normal_(self.input_scale)
            backbone_layers: list[Any] = []
            in_features = input_dim
            for _ in range(spec.hidden_layers):
                backbone_layers.extend(
                    [
                        nn.Linear(in_features, spec.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(spec.dropout),
                    ]
                )
                in_features = spec.hidden_size
            self.backbone = nn.Sequential(*backbone_layers)
            self.head_weight = nn.Parameter(torch.empty(spec.k, in_features))
            self.head_bias = nn.Parameter(torch.empty(spec.k))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.head_weight, -bound, bound)
            nn.init.uniform_(self.head_bias, -bound, bound)

        def forward(self, x: Any) -> Any:
            # x: (B, D) -> member representations: (B, K, D)
            representation = x.unsqueeze(1) * self.input_scale.unsqueeze(0)
            representation = self.backbone(representation)
            return (
                (representation * self.head_weight.unsqueeze(0)).sum(dim=-1)
                + self.head_bias.unsqueeze(0)
            )

    return MiniEnsemble()


def _cache_training_matrix(
    torch: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    device: Any,
) -> tuple[Any, Any, Any, bool]:
    if device.type == "cuda":
        available, _total = torch.cuda.mem_get_info(device)
        cache_on_device = matrix.nbytes + target.nbytes < int(available * 0.65)
    else:
        cache_on_device = True
    data_device = device if cache_on_device else torch.device("cpu")
    X_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=data_device)
    y_tensor = torch.as_tensor(target, dtype=torch.float32, device=data_device)
    return X_tensor, y_tensor, data_device, cache_on_device


def _run_epoch(
    torch: Any,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    X_tensor: Any,
    y_tensor: Any,
    data_device: Any,
    device: Any,
    spec: TabMMiniSpec,
    use_amp: bool,
) -> float:
    model.train()
    n_rows = int(X_tensor.shape[0])
    permutation = torch.randperm(n_rows, device=data_device)
    total_loss = 0.0
    observed = 0
    for left in range(0, n_rows, spec.batch_size):
        indices = permutation[left : left + spec.batch_size]
        x_batch = X_tensor[indices]
        y_batch = y_tensor[indices]
        if data_device.type != device.type:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            member_predictions = model(x_batch)
            # Crucial TabM detail: MSE for each member, then mean. Never train
            # the loss of member_predictions.mean(dim=1).
            loss = torch.square(
                member_predictions - y_batch.unsqueeze(1)
            ).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_rows = int(y_batch.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_rows
        observed += batch_rows
    return total_loss / observed


def _predict_members(
    torch: Any,
    model: Any,
    matrix: np.ndarray,
    *,
    device: Any,
    batch_size: int,
    k: int,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    result = np.empty((matrix.shape[0], k), dtype=np.float32)
    with torch.no_grad():
        for left in range(0, matrix.shape[0], batch_size):
            right = min(left + batch_size, matrix.shape[0])
            batch = torch.as_tensor(matrix[left:right], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(batch)
            result[left:right] = prediction.float().cpu().numpy()
    return result


def _make_preprocessor(names: Sequence[str], kinds: Sequence[str]) -> RobustPreprocessor:
    return RobustPreprocessor(
        feature_names=list(names),
        feature_kinds=list(kinds),
        clip=8.0,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )


def _train_one_fold(
    torch: Any,
    *,
    X_raw: np.ndarray,
    names: Sequence[str],
    kinds: Sequence[str],
    months: np.ndarray,
    target: np.ndarray,
    fold: Any,
    spec: TabMMiniSpec,
    device: Any,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    prediction_path = OUTPUT_ROOT / f"{fold.name.lower()}_predictions.npz"
    summary_path = OUTPUT_ROOT / f"{fold.name.lower()}_summary.json"
    if prediction_path.exists() and summary_path.exists() and not overwrite:
        cached = np.load(prediction_path, allow_pickle=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = summary["selection_epochs"] + summary.get("refit_epochs", [])
        return cached["prediction"].astype(np.float64), summary, history

    inner_start = fold.train_end - spec.inner_val_months + 1
    if inner_start <= fold.train_start:
        raise ValueError(f"{fold.name}: inner validation consumes training")
    fit_slice = _month_slice(months, fold.train_start, inner_start - 1)
    inner_slice = _month_slice(months, inner_start, fold.train_end)
    outer_train_slice = _month_slice(months, fold.train_start, fold.train_end)
    outer_slice = _month_slice(months, fold.validation_start, fold.validation_end)
    y_outer = target[outer_slice]
    use_amp = device.type == "cuda"
    fold_seed = spec.seed + 100 + int(fold.name[-1])

    selection_transform_started = time.perf_counter()
    selection_preprocessor = _make_preprocessor(names, kinds)
    selection_preprocessor.fit(X_raw[fit_slice])
    X_fit = np.ascontiguousarray(selection_preprocessor.transform(X_raw[fit_slice]))
    X_inner = np.ascontiguousarray(
        selection_preprocessor.transform(X_raw[inner_slice])
    )
    _smooth_clip_inplace(X_fit, spec.smooth_clip_scale)
    _smooth_clip_inplace(X_inner, spec.smooth_clip_scale)
    selection_transform_seconds = time.perf_counter() - selection_transform_started

    y_fit = target[fit_slice]
    y_inner = target[inner_slice]
    selection_target_mean = float(np.mean(y_fit))
    selection_target_std = float(np.std(y_fit))
    y_fit_standard = ((y_fit - selection_target_mean) / selection_target_std).astype(
        np.float32
    )

    _set_determinism(torch, fold_seed)
    selection_model = _build_tabm_mini(torch, X_fit.shape[1], spec).to(device)
    selection_optimizer = torch.optim.AdamW(
        selection_model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    selection_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    X_fit_tensor, y_fit_tensor, data_device, cache_on_device = _cache_training_matrix(
        torch, X_fit, y_fit_standard, device
    )
    if cache_on_device:
        del X_fit, y_fit_standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_epoch = 0
    best_inner_cosine = -np.inf
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    selection_rows: list[dict[str, Any]] = []
    selection_training_started = time.perf_counter()
    for epoch in range(1, spec.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(
            torch,
            model=selection_model,
            optimizer=selection_optimizer,
            scaler=selection_scaler,
            X_tensor=X_fit_tensor,
            y_tensor=y_fit_tensor,
            data_device=data_device,
            device=device,
            spec=spec,
            use_amp=use_amp,
        )
        inner_members_standard = _predict_members(
            torch,
            selection_model,
            X_inner,
            device=device,
            batch_size=spec.inference_batch_size,
            k=spec.k,
            use_amp=use_amp,
        )
        inner_members = (
            inner_members_standard.astype(np.float64) * selection_target_std
            + selection_target_mean
        )
        inner_ensemble = inner_members.mean(axis=1)
        inner_cosine = cosine_score(y_inner, inner_ensemble)
        improved = inner_cosine > best_inner_cosine + 1e-7
        if improved:
            best_epoch = epoch
            best_inner_cosine = inner_cosine
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in selection_model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        row = {
            "fold": fold.name,
            "stage": "selection",
            "epoch": epoch,
            "train_individual_member_mse_standardized": train_loss,
            "inner_ensemble_cosine": inner_cosine,
            "inner_member_cosine_mean": float(
                np.mean(
                    [cosine_score(y_inner, inner_members[:, j]) for j in range(spec.k)]
                )
            ),
            "improved": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        selection_rows.append(row)
        print(
            f"{fold.name} TabM-mini epoch {epoch:03d}: member_mse={train_loss:.6f} "
            f"inner_ensemble_cos={inner_cosine:.6f} best={best_inner_cosine:.6f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if epoch >= spec.min_epochs and no_improvement >= spec.patience:
            break
    if best_state is None:
        raise RuntimeError(f"{fold.name}: no TabM-mini checkpoint selected")
    selection_training_seconds = time.perf_counter() - selection_training_started
    selection_features = int(X_fit_tensor.shape[1])
    del (
        selection_model,
        selection_optimizer,
        selection_scaler,
        X_fit_tensor,
        y_fit_tensor,
        X_inner,
        selection_preprocessor,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Refit from scratch on every outer-training month for selected epochs.
    refit_transform_started = time.perf_counter()
    refit_preprocessor = _make_preprocessor(names, kinds)
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
    y_refit_standard = ((y_refit - refit_target_mean) / refit_target_std).astype(
        np.float32
    )
    refit_features = int(X_refit.shape[1])
    _set_determinism(torch, fold_seed)
    refit_model = _build_tabm_mini(torch, refit_features, spec).to(device)
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    refit_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    X_refit_tensor, y_refit_tensor, refit_device, refit_cached = _cache_training_matrix(
        torch, X_refit, y_refit_standard, device
    )
    if refit_cached:
        del X_refit, y_refit_standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    refit_rows: list[dict[str, Any]] = []
    refit_training_started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(
            torch,
            model=refit_model,
            optimizer=refit_optimizer,
            scaler=refit_scaler,
            X_tensor=X_refit_tensor,
            y_tensor=y_refit_tensor,
            data_device=refit_device,
            device=device,
            spec=spec,
            use_amp=use_amp,
        )
        row = {
            "fold": fold.name,
            "stage": "refit",
            "epoch": epoch,
            "train_individual_member_mse_standardized": train_loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        refit_rows.append(row)
        print(
            f"{fold.name} TabM-mini refit {epoch:03d}/{best_epoch:03d}: "
            f"member_mse={train_loss:.6f} seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
    refit_training_seconds = time.perf_counter() - refit_training_started

    outer_members_standard = _predict_members(
        torch,
        refit_model,
        X_outer,
        device=device,
        batch_size=spec.inference_batch_size,
        k=spec.k,
        use_amp=use_amp,
    )
    outer_members = (
        outer_members_standard.astype(np.float64) * refit_target_std
        + refit_target_mean
    )
    outer_prediction = outer_members.mean(axis=1)
    member_cosines = np.array(
        [cosine_score(y_outer, outer_members[:, j]) for j in range(spec.k)]
    )
    member_correlation = np.corrcoef(outer_members, rowvar=False)
    mean_off_diagonal_correlation = float(
        (member_correlation.sum() - spec.k) / (spec.k * (spec.k - 1))
    )

    summary: dict[str, Any] = {
        "fold": fold.name,
        "outer_train_months": [fold.train_start, fold.train_end],
        "inner_fit_months": [fold.train_start, inner_start - 1],
        "inner_validation_months": [inner_start, fold.train_end],
        "outer_validation_months": [fold.validation_start, fold.validation_end],
        "selection_fit_rows": int(fit_slice.stop - fit_slice.start),
        "inner_validation_rows": int(inner_slice.stop - inner_slice.start),
        "refit_rows": int(outer_train_slice.stop - outer_train_slice.start),
        "outer_validation_rows": int(outer_slice.stop - outer_slice.start),
        "raw_features": int(X_raw.shape[1]),
        "selection_transformed_features": selection_features,
        "refit_transformed_features": refit_features,
        "device": str(device),
        "amp": use_amp,
        "selection_cached_on_device": cache_on_device,
        "refit_cached_on_device": refit_cached,
        "best_epoch": best_epoch,
        "best_inner_ensemble_cosine": best_inner_cosine,
        "refit_from_scratch_on_all_outer_train_months": True,
        "loss_contract": "mean of k individual-member MSE values",
        "inference_contract": "arithmetic mean of k member predictions",
        "outer_ensemble_cosine_raw": cosine_score(y_outer, outer_prediction),
        "outer_ensemble_cosine_power": cosine_score(
            y_outer, _power_view(outer_prediction, spec.prediction_power)
        ),
        "outer_member_cosine_mean": float(member_cosines.mean()),
        "outer_member_cosine_min": float(member_cosines.min()),
        "outer_member_cosine_max": float(member_cosines.max()),
        "outer_member_mean_pairwise_correlation": mean_off_diagonal_correlation,
        "selection_transform_seconds": selection_transform_seconds,
        "selection_training_seconds": selection_training_seconds,
        "refit_transform_seconds": refit_transform_seconds,
        "refit_training_seconds": refit_training_seconds,
        "selection_epochs": selection_rows,
        "refit_epochs": refit_rows,
    }
    np.savez_compressed(
        prediction_path,
        row_indices=np.arange(outer_slice.start, outer_slice.stop, dtype=np.int64),
        months=months[outer_slice],
        target=y_outer,
        prediction=outer_prediction,
        member_predictions=outer_members.astype(np.float32),
    )
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in refit_model.state_dict().items()
            },
            "input_dim": refit_features,
            "target_mean": refit_target_mean,
            "target_std": refit_target_std,
            "selected_epoch": best_epoch,
            "spec": asdict(spec),
            "fold": fold.name,
        },
        OUTPUT_ROOT / f"{fold.name.lower()}_checkpoint.pt",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
    return outer_prediction, summary, selection_rows + refit_rows


def _compare_candidates(
    target: np.ndarray,
    months: np.ndarray,
    row_indices: np.ndarray,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    baseline = np.load(BASELINE_OOF, allow_pickle=False)
    if not np.array_equal(row_indices, baseline["row_indices"]):
        raise ValueError("TabM OOF rows do not align with LightGBM OOF")
    if not np.array_equal(months, baseline["months"]):
        raise ValueError("TabM OOF months do not align with LightGBM OOF")
    if not np.array_equal(target, baseline["target"]):
        raise ValueError("TabM OOF targets do not align with LightGBM OOF")

    views: dict[str, np.ndarray] = {
        "lightgbm_slow": baseline["slow"].astype(np.float64),
        "tabm_mini_raw": prediction.astype(np.float64),
    }
    for power in (1.0, 1.1, 1.2, 1.3):
        views[f"tabm_mini_power_{power:g}"] = _power_view(prediction, power)
    selection_mask = months <= DEVELOPMENT_FOLDS[1].validation_end
    audit_mask = months >= DEVELOPMENT_FOLDS[2].validation_start
    best_tabm = max(
        (name for name in views if name.startswith("tabm")),
        key=lambda name: cosine_score(target[selection_mask], views[name][selection_mask]),
    )
    gbdt = rms_normalize(views["lightgbm_slow"])
    tabm = rms_normalize(views[best_tabm])
    for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
        views[f"blend_tabm_{weight:g}"] = rms_normalize(
            (1.0 - weight) * gbdt + weight * tabm
        )

    rows: list[dict[str, Any]] = []
    for name, candidate in views.items():
        monthly = monthly_diagnostics(target, candidate, months)
        row = summarize_monthly_diagnostics(
            monthly, cosine_score(target, candidate)
        ).to_dict()
        row["candidate"] = name
        row["selection_cosine_dev1_dev2"] = cosine_score(
            target[selection_mask], candidate[selection_mask]
        )
        row["audit_cosine_dev3"] = cosine_score(
            target[audit_mask], candidate[audit_mask]
        )
        for fold in DEVELOPMENT_FOLDS:
            mask = (months >= fold.validation_start) & (months <= fold.validation_end)
            row[f"{fold.name.lower()}_cosine"] = cosine_score(
                target[mask], candidate[mask]
            )
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(
            "selection_cosine_dev1_dev2", ascending=False, ignore_index=True
        ),
        views,
    )


def evaluate(spec: TabMMiniSpec, fold_names: Sequence[str], overwrite: bool) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable for TabM-mini") from exc

    _set_determinism(torch, spec.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = [fold for fold in DEVELOPMENT_FOLDS if fold.name in set(fold_names)]
    if len(selected) != len(set(fold_names)):
        raise ValueError("unknown or duplicate development fold")

    months, target = _load_labels()
    features = materialize_feature_set("train", spec.feature_set)
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in selected:
        prediction, summary, fold_history = _train_one_fold(
            torch,
            X_raw=features.matrix,
            names=features.names,
            kinds=features.kinds,
            months=months,
            target=target,
            fold=fold,
            spec=spec,
            device=device,
            overwrite=overwrite,
        )
        validation_slice = _month_slice(months, fold.validation_start, fold.validation_end)
        indices.append(np.arange(validation_slice.start, validation_slice.stop, dtype=np.int64))
        predictions.append(prediction)
        summaries.append(summary)
        history.extend(fold_history)

    oof_rows = np.concatenate(indices)
    oof_prediction = np.concatenate(predictions)
    oof_target = target[oof_rows]
    oof_months = months[oof_rows]
    np.savez_compressed(
        OUTPUT_ROOT / "tabm_mini_development_oof.npz",
        row_indices=oof_rows,
        months=oof_months,
        target=oof_target,
        prediction=oof_prediction,
    )
    pd.DataFrame(history).to_csv(OUTPUT_ROOT / "training_history.csv", index=False)

    baseline_rows = np.load(BASELINE_OOF, allow_pickle=False)["row_indices"]
    complete = len(selected) == len(DEVELOPMENT_FOLDS) and np.array_equal(
        oof_rows, baseline_rows
    )
    ranking: pd.DataFrame | None = None
    if complete:
        ranking, views = _compare_candidates(
            oof_target, oof_months, oof_rows, oof_prediction
        )
        ranking.to_csv(OUTPUT_ROOT / "candidate_comparison.csv", index=False)
        np.savez_compressed(
            OUTPUT_ROOT / "candidate_views.npz",
            row_indices=oof_rows,
            months=oof_months,
            target=oof_target,
            **views,
        )
        best_candidate = str(ranking.iloc[0]["candidate"])
    else:
        best_candidate = "not_available_partial_fold_run"

    payload: dict[str, Any] = {
        "experiment": "tabm_mini_strict_chronological_challenger",
        "spec": asdict(spec),
        "selected_folds": [fold.name for fold in selected],
        "complete_development_oof": complete,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "folds": summaries,
        "partial_or_pooled_tabm_cosine": cosine_score(oof_target, oof_prediction),
        "partial_or_pooled_power_cosine": cosine_score(
            oof_target, _power_view(oof_prediction, spec.prediction_power)
        ),
        "best_candidate": best_candidate,
        "ranking": None if ranking is None else ranking.to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - started,
        "source_references": [
            "https://github.com/yandex-research/tabm",
            "https://arxiv.org/abs/2410.24210",
        ],
    }
    summary_path = OUTPUT_ROOT / "tabm_mini_challenger_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_sha256"] = _sha256(summary_path)
    print(
        f"TabM-mini challenger ({','.join(fold_names)}): "
        f"cosine={payload['partial_or_pooled_tabm_cosine']:.6f}, device={device}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    defaults = TabMMiniSpec()
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TabMMiniSpec(
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
    )
    evaluate(spec, args.folds, args.overwrite)


if __name__ == "__main__":
    main()

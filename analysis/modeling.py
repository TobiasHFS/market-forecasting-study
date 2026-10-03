"""Leakage-conscious modeling and validation utilities for the competition.

This module deliberately separates *development* model selection from the sealed
months 59--70 audit.  Its default workflow is:

1. Engineer numeric features without ``sample_id``.
2. Compare a small, predeclared candidate set with
   :func:`run_development_oof` or :func:`compare_development_candidates`.
3. Freeze the feature set, hyperparameters, prediction-view construction, and
   blend weight.
4. Call :func:`run_sealed_audit` once, with ``confirm_frozen=True``.

``RobustPreprocessor`` expects already engineered, sample-level numeric
features.  It can apply a few explicitly declared monotone transforms, but it
does not attempt feature discovery.  All imputation, transform scales, robust
centers, robust scales, missing-indicator selection, and clipping thresholds
are fitted on the training slice only.

The competition score and every prediction-normalization helper here are
*uncentered*.  In particular, no prediction vector is z-scored and no monthly
normalization is performed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


ArrayLike = np.ndarray | pd.DataFrame
EstimatorFactory = Callable[[], BaseEstimator]


# A deliberately small grid.  Any expansion should be justified before seeing
# the sealed audit, and compared only with development OOF predictions.
# Per-observation penalties.  The fitted sklearn alpha is lambda times the
# effective training weight, so regularization strength does not silently fade
# as expanding folds acquire hundreds of thousands more rows.
RIDGE_ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BLEND_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ChronologicalFold:
    """Inclusive month boundaries for one expanding-window split."""

    name: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    sealed: bool = False

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError("train_start must not exceed train_end")
        if self.validation_start > self.validation_end:
            raise ValueError("validation_start must not exceed validation_end")
        if self.train_end >= self.validation_start:
            raise ValueError("training must end before validation starts")


DEVELOPMENT_FOLDS: tuple[ChronologicalFold, ...] = (
    ChronologicalFold("Dev1", 0, 22, 23, 34),
    ChronologicalFold("Dev2", 0, 34, 35, 46),
    ChronologicalFold("Dev3", 0, 46, 47, 58),
)

SEALED_AUDIT_FOLD = ChronologicalFold(
    "SealedAudit", 0, 58, 59, 70, sealed=True
)


def rms(values: Sequence[float] | np.ndarray) -> float:
    """Return the uncentered root mean square."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("RMS input contains NaN or infinity")
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def cosine_score(
    y_true: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> float:
    """Compute the exact uncentered cosine used by the competition.

    A zero target or prediction norm is an invalid scoring vector and raises a
    ``ValueError`` rather than silently inventing a score.
    """

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: target {y.shape}, prediction {p.shape}")
    if y.size == 0:
        raise ValueError("cannot score an empty vector")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("cosine inputs contain NaN or infinity")
    denominator = np.sqrt(np.dot(y, y) * np.dot(p, p))
    if denominator <= 0.0:
        raise ValueError("cosine is undefined for a zero-norm vector")
    return float(np.dot(y, p) / denominator)


def rms_normalize(
    prediction: Sequence[float] | np.ndarray,
    *,
    target_rms: float = 1.0,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Scale one complete prediction vector to an uncentered RMS.

    Supply the entire OOF, outer-validation, stress-test, or test vector at
    once.  Do not call this separately by month or leaderboard partition.
    """

    if not np.isfinite(target_rms) or target_rms <= 0.0:
        raise ValueError("target_rms must be finite and positive")
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    scale = rms(p)
    if scale <= epsilon:
        raise ValueError("cannot RMS-normalize a zero (or near-zero) vector")
    return p * (target_rms / scale)


def assemble_prediction_views(
    view_predictions: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    renormalize: bool = True,
) -> np.ndarray:
    """Unit-RMS normalize views, take a nonnegative average, then renormalize.

    ``view_predictions`` has shape ``(n_samples, n_views)``.  Normalization is
    once per full column, not per fold-month.  Equal weights are the default.
    """

    matrix = np.asarray(view_predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("view_predictions must be a nonempty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("view_predictions contains NaN or infinity")
    if weights is None:
        weight = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight.shape[0] != matrix.shape[1]:
            raise ValueError("weights length does not match number of views")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("view weights must be finite and nonnegative")
        if weight.sum() <= 0.0:
            raise ValueError("at least one view weight must be positive")
        weight = weight / weight.sum()
    normalized = np.column_stack(
        [rms_normalize(matrix[:, column]) for column in range(matrix.shape[1])]
    )
    combined = normalized @ weight
    return rms_normalize(combined) if renormalize else combined


_VALID_FEATURE_KINDS = {
    "identity",
    "bounded",
    "log1p_nonnegative",
    "asinh_signed",
}


class RobustPreprocessor(TransformerMixin, BaseEstimator):
    """Train-only finite handling, robust scaling, clipping, and transforms.

    Parameters
    ----------
    feature_kinds:
        Either a sequence aligned with columns or, for DataFrames, a mapping
        from column name to one of ``identity``, ``bounded``,
        ``log1p_nonnegative``, or ``asinh_signed``.  Unspecified columns use
        ``identity``.  Negative values of a ``log1p_nonnegative`` feature are
        treated as missing.  ``asinh_signed`` uses a train-fitted magnitude
        scale before the robust standardization.
    clip:
        Symmetric clipping threshold after median/MAD standardization.
    add_missing_indicators:
        Append an indicator for each column that had at least one invalid value
        in the training slice.  This selection is itself train-only.
    output_dtype:
        ``numpy.float32`` is recommended for the full competition matrix.

    Notes
    -----
    Input columns should be engineered sample-level quantities; identifiers and
    month labels must not be included as features.  MAD is multiplied by
    1.4826.  Degenerate columns fall back to IQR/1.349, then standard deviation,
    then a scale of one.
    """

    def __init__(
        self,
        *,
        feature_names: Sequence[str] | None = None,
        feature_kinds: Mapping[str, str] | Sequence[str] | None = None,
        clip: float = 8.0,
        add_missing_indicators: bool = True,
        output_dtype: np.dtype | type = np.float32,
    ) -> None:
        self.feature_names = feature_names
        self.feature_kinds = feature_kinds
        self.clip = clip
        self.add_missing_indicators = add_missing_indicators
        self.output_dtype = output_dtype

    def _column_names(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValueError("feature DataFrame columns must be unique")
            if not all(isinstance(name, str) for name in X.columns):
                raise TypeError("feature DataFrame columns must all be strings")
            names = np.asarray(X.columns, dtype=object)
            forbidden = {"sample_id", "month", "target"}.intersection(names.tolist())
            if forbidden:
                raise ValueError(
                    "identifier/label columns must not enter the feature matrix: "
                    f"{sorted(forbidden)}"
                )
            if self.feature_names is not None:
                declared = np.asarray(tuple(self.feature_names), dtype=object)
                if not np.array_equal(names, declared):
                    raise ValueError(
                        "DataFrame columns do not exactly match declared feature_names"
                    )
            return names
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if self.feature_names is None:
            raise ValueError(
                "ndarray input requires explicit feature_names so identifiers "
                "and labels cannot enter anonymously"
            )
        names = np.asarray(tuple(self.feature_names), dtype=object)
        if names.shape != (array.shape[1],):
            raise ValueError("feature_names length does not match X columns")
        if not all(isinstance(name, str) for name in names):
            raise TypeError("feature_names must contain only strings")
        if len(set(names.tolist())) != len(names):
            raise ValueError("feature_names must be unique")
        forbidden = {"sample_id", "month", "target"}.intersection(names.tolist())
        if forbidden:
            raise ValueError(
                "identifier/label columns must not enter the feature matrix: "
                f"{sorted(forbidden)}"
            )
        return names

    @staticmethod
    def _column(X: ArrayLike, index: int, name: str) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if name not in X.columns:
                raise ValueError(f"transform input is missing feature {name!r}")
            raw = X[name].to_numpy(copy=False)
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be two-dimensional")
            raw = array[:, index]
        try:
            return np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"feature {name!r} is not numeric") from exc

    def _resolve_kinds(self, names: np.ndarray) -> np.ndarray:
        if self.feature_kinds is None:
            kinds = ["identity"] * len(names)
        elif isinstance(self.feature_kinds, Mapping):
            if not all(isinstance(name, str) for name in names):
                raise TypeError("mapping feature_kinds requires named columns")
            unknown = set(self.feature_kinds).difference(set(names.tolist()))
            if unknown:
                raise ValueError(f"feature_kinds contains unknown columns: {unknown}")
            kinds = [self.feature_kinds.get(str(name), "identity") for name in names]
        else:
            kinds = list(self.feature_kinds)
            if len(kinds) != len(names):
                raise ValueError("feature_kinds length does not match X columns")
        invalid = set(kinds).difference(_VALID_FEATURE_KINDS)
        if invalid:
            raise ValueError(f"unknown feature transform kinds: {invalid}")
        return np.asarray(kinds, dtype=object)

    @staticmethod
    def _fit_asinh_scale(raw: np.ndarray, valid: np.ndarray) -> float:
        magnitude = np.abs(raw[valid])
        if magnitude.size == 0:
            return 1.0
        scale = float(np.median(magnitude))
        if not np.isfinite(scale) or scale <= 1e-12:
            nonzero = magnitude[magnitude > 1e-12]
            scale = float(np.median(nonzero)) if nonzero.size else 1.0
        return max(scale, 1e-12)

    @staticmethod
    def _transform_valid(
        raw_valid: np.ndarray, kind: str, transform_scale: float
    ) -> np.ndarray:
        if kind == "log1p_nonnegative":
            return np.log1p(raw_valid)
        if kind == "asinh_signed":
            return np.arcsinh(raw_valid / transform_scale)
        return raw_valid

    def fit(self, X: ArrayLike, y: np.ndarray | None = None) -> "RobustPreprocessor":
        del y
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("clip must be finite and positive")
        names = self._column_names(X)
        if len(X) == 0 or len(names) == 0:
            raise ValueError("X must contain at least one row and one column")
        kinds = self._resolve_kinds(names)
        medians = np.zeros(len(names), dtype=np.float64)
        scales = np.ones(len(names), dtype=np.float64)
        transform_scales = np.ones(len(names), dtype=np.float64)
        missing = np.zeros(len(names), dtype=bool)

        for column, (name, kind) in enumerate(zip(names, kinds, strict=True)):
            raw = self._column(X, column, str(name))
            valid = np.isfinite(raw)
            if kind == "log1p_nonnegative":
                valid &= raw >= 0.0
            missing[column] = not bool(np.all(valid))
            if kind == "asinh_signed":
                transform_scales[column] = self._fit_asinh_scale(raw, valid)
            values = self._transform_valid(
                raw[valid], str(kind), transform_scales[column]
            )
            if values.size == 0:
                medians[column], scales[column] = 0.0, 1.0
                continue
            center = float(np.median(values))
            scale = 1.4826 * float(np.median(np.abs(values - center)))
            if not np.isfinite(scale) or scale <= 1e-12:
                q25, q75 = np.percentile(values, [25.0, 75.0])
                scale = float((q75 - q25) / 1.349)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(values, dtype=np.float64))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            medians[column], scales[column] = center, scale

        self.n_features_in_ = len(names)
        self.feature_names_in_ = names
        self.feature_kinds_ = kinds
        self.medians_ = medians
        self.scales_ = scales
        self.transform_scales_ = transform_scales
        self.missing_indicator_mask_ = missing if self.add_missing_indicators else np.zeros_like(missing)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=("medians_", "scales_", "transform_scales_"),
        )
        if len(X) == 0:
            raise ValueError("X must contain at least one row")
        if not isinstance(X, pd.DataFrame):
            array = np.asarray(X)
            if array.ndim != 2 or array.shape[1] != self.n_features_in_:
                raise ValueError("transform input has the wrong number of columns")
        elif not np.array_equal(
            np.asarray(X.columns, dtype=object), self.feature_names_in_
        ):
            raise ValueError(
                "transform DataFrame columns must exactly match the fitted schema"
            )
        n_rows = len(X)
        n_indicators = int(self.missing_indicator_mask_.sum())
        output = np.empty(
            (n_rows, self.n_features_in_ + n_indicators), dtype=self.output_dtype
        )
        if isinstance(X, pd.DataFrame):
            source = X.loc[:, self.feature_names_in_.tolist()].to_numpy(copy=False)
        else:
            source = np.asarray(X)
        log_columns = np.flatnonzero(self.feature_kinds_ == "log1p_nonnegative")
        asinh_columns = np.flatnonzero(self.feature_kinds_ == "asinh_signed")
        indicator_columns = np.flatnonzero(self.missing_indicator_mask_)

        # Row chunks keep writes contiguous and cap temporary float64/bool
        # memory.  Column-wise writes are prohibitively cache-inefficient for a
        # million-row, few-hundred-column matrix.
        target_temporary_bytes = 192 * 1024 * 1024
        rows_per_chunk = max(
            1_000,
            min(n_rows, target_temporary_bytes // max(16 * self.n_features_in_, 1)),
        )
        for start in range(0, n_rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_rows)
            transformed = np.asarray(source[start:stop], dtype=np.float64).copy()
            valid = np.isfinite(transformed)
            if log_columns.size:
                log_values = transformed[:, log_columns]
                log_valid = np.isfinite(log_values) & (log_values >= 0.0)
                valid[:, log_columns] = log_valid
                log_values[~log_valid] = 0.0
                transformed[:, log_columns] = np.log1p(log_values)
            if asinh_columns.size:
                asinh_values = transformed[:, asinh_columns]
                transformed[:, asinh_columns] = np.arcsinh(
                    asinh_values / self.transform_scales_[asinh_columns]
                )
            if not np.all(valid):
                medians = np.broadcast_to(self.medians_, transformed.shape)
                transformed[~valid] = medians[~valid]
            transformed -= self.medians_
            transformed /= self.scales_
            np.clip(transformed, -self.clip, self.clip, out=transformed)
            output[start:stop, : self.n_features_in_] = transformed
            if indicator_columns.size:
                output[start:stop, self.n_features_in_ :] = (~valid[:, indicator_columns])
        return output

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, attributes=("feature_names_in_",))
        if input_features is not None:
            supplied = np.asarray(input_features, dtype=object)
            if not np.array_equal(supplied, self.feature_names_in_):
                raise ValueError("input_features does not match fitted feature order")
        names = self.feature_names_in_.astype(object).tolist()
        names.extend(
            f"{name}__missing"
            for name, include in zip(
                self.feature_names_in_, self.missing_indicator_mask_, strict=True
            )
            if include
        )
        return np.asarray(names, dtype=object)


class PerObservationRidge(RegressorMixin, BaseEstimator):
    """Ridge with a penalty stable across fold sample sizes.

    Scikit-learn minimizes ``||y-Xb||^2 + alpha||b||^2``.  Here ``penalty`` is
    the per-observation lambda and the fitted absolute alpha is
    ``penalty * sum(sample_weight)`` (or ``penalty * n`` without weights).
    """

    def __init__(
        self,
        penalty: float = 1e-2,
        *,
        fit_intercept: bool = True,
        solver: str = "lsqr",
        tol: float = 1e-4,
        max_iter: int = 2_000,
    ) -> None:
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "PerObservationRidge":
        if not np.isfinite(self.penalty) or self.penalty < 0.0:
            raise ValueError("penalty must be finite and nonnegative")
        effective_n = (
            float(np.sum(sample_weight, dtype=np.float64))
            if sample_weight is not None
            else float(len(y))
        )
        if not np.isfinite(effective_n) or effective_n <= 0.0:
            raise ValueError("effective training weight must be positive")
        self.alpha_ = float(self.penalty) * effective_n
        self.model_ = Ridge(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=("model_", "alpha_"))
        return self.model_.predict(X)


def make_ridge_estimator(
    alpha: float,
    *,
    feature_names: Sequence[str] | None = None,
    feature_kinds: Mapping[str, str] | Sequence[str] | None = None,
    clip: float = 8.0,
    add_missing_indicators: bool = True,
) -> Pipeline:
    """Create a robustly preprocessed per-observation Ridge candidate."""

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    return Pipeline(
        steps=(
            (
                "preprocess",
                RobustPreprocessor(
                    feature_names=feature_names,
                    feature_kinds=feature_kinds,
                    clip=clip,
                    add_missing_indicators=add_missing_indicators,
                ),
            ),
            (
                "model",
                PerObservationRidge(penalty=float(alpha)),
            ),
        )
    )


@dataclass(frozen=True)
class LightGBMSpec:
    """Shallow, strongly regularized, fixed-iteration LightGBM specification."""

    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 1_000
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.70
    reg_alpha: float = 1.0
    reg_lambda: float = 20.0
    max_bin: int = 63
    objective: str = "regression_l2"
    random_state: int = 20260823
    n_jobs: int = -1
    verbosity: int = -1


def make_lightgbm_estimator(
    spec: LightGBMSpec | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    feature_kinds: Mapping[str, str] | Sequence[str] | None = None,
    clip: float = 8.0,
    add_missing_indicators: bool = True,
) -> Pipeline:
    """Create a robust fixed-iteration LightGBM candidate.

    Fixed iterations avoid using the evaluated outer validation fold for early
    stopping.  If early stopping is desired, it belongs inside a nested inner
    split, not in :func:`run_development_oof`.
    """

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "LightGBM is optional; install the 'lightgbm' package to use this candidate"
        ) from exc
    params = asdict(spec or LightGBMSpec())
    return Pipeline(
        steps=(
            (
                "preprocess",
                RobustPreprocessor(
                    feature_names=feature_names,
                    feature_kinds=feature_kinds,
                    clip=clip,
                    add_missing_indicators=add_missing_indicators,
                ),
            ),
            ("model", LGBMRegressor(**params)),
        )
    )


@dataclass(frozen=True)
class RecencyWeighting:
    """A predeclared exponential half-life, in whole months.

    Leave this disabled (``None`` in the evaluation functions) unless the
    half-life was declared before inspecting candidate fold results.  If more
    than one half-life is compared, that comparison is a hyperparameter search
    and must remain entirely inside development folds.
    """

    half_life_months: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.half_life_months) or self.half_life_months <= 0.0:
            raise ValueError("half_life_months must be finite and positive")

    def weights(self, train_months: np.ndarray, train_end: int) -> np.ndarray:
        month = np.asarray(train_months, dtype=np.float64).reshape(-1)
        if month.size == 0 or not np.all(np.isfinite(month)):
            raise ValueError("train_months must be a nonempty finite vector")
        age = np.maximum(float(train_end) - month, 0.0)
        weight = np.exp(log(0.5) * age / self.half_life_months)
        return weight / np.mean(weight)


def _slice_rows(X: ArrayLike, indices: np.ndarray) -> ArrayLike:
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else np.asarray(X)[indices]


def fold_indices(months: Sequence[int] | np.ndarray, fold: ChronologicalFold) -> tuple[np.ndarray, np.ndarray]:
    """Return exact row indices for a fold and validate that both sides exist."""

    month = np.asarray(months).reshape(-1)
    if month.size == 0 or not np.all(np.isfinite(month)):
        raise ValueError("months must be a nonempty finite vector")
    if not np.all(month == np.floor(month)):
        raise ValueError("months must be integer-valued")
    train = np.flatnonzero((month >= fold.train_start) & (month <= fold.train_end))
    valid = np.flatnonzero(
        (month >= fold.validation_start) & (month <= fold.validation_end)
    )
    if train.size == 0:
        raise ValueError(f"{fold.name} has no training rows")
    if valid.size == 0:
        raise ValueError(f"{fold.name} has no validation rows")
    if np.max(month[train]) >= np.min(month[valid]):
        raise AssertionError(f"{fold.name} is not chronologically ordered")
    return train, valid


def _fit_with_optional_weight(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> BaseEstimator:
    if sample_weight is None:
        estimator.fit(X, y)
    elif isinstance(estimator, Pipeline):
        final_step = estimator.steps[-1][0]
        estimator.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
    else:
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


@dataclass(frozen=True)
class OOFResult:
    """Predictions from disjoint chronological validation blocks."""

    row_indices: np.ndarray
    months: np.ndarray
    y_true: np.ndarray
    prediction: np.ndarray
    fold_names: np.ndarray

    @property
    def pooled_cosine(self) -> float:
        """Cosine after concatenating every included validation observation."""

        return cosine_score(self.y_true, self.prediction)

    @property
    def monthly(self) -> pd.DataFrame:
        return monthly_diagnostics(self.y_true, self.prediction, self.months)

    @property
    def summary(self) -> pd.Series:
        return summarize_monthly_diagnostics(self.monthly, self.pooled_cosine)


def _run_fold_set(
    X: ArrayLike,
    y: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    estimator_factory: EstimatorFactory,
    folds: Sequence[ChronologicalFold],
    *,
    recency_weighting: RecencyWeighting | None = None,
    allow_sealed: bool = False,
) -> OOFResult:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    month = np.asarray(months).reshape(-1)
    if len(X) != target.size or month.size != target.size:
        raise ValueError("X, y, and months must have the same number of rows")
    if not np.all(np.isfinite(target)):
        raise ValueError("training target contains NaN or infinity")
    if any(fold.sealed for fold in folds) and not allow_sealed:
        raise RuntimeError("sealed fold access is not allowed in development evaluation")

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_months: list[np.ndarray] = []
    all_names: list[np.ndarray] = []

    for fold in folds:
        train_index, validation_index = fold_indices(month, fold)
        estimator = estimator_factory()
        if estimator is None:
            raise TypeError("estimator_factory returned None")
        weight = (
            recency_weighting.weights(month[train_index], fold.train_end)
            if recency_weighting is not None
            else None
        )
        _fit_with_optional_weight(
            estimator,
            _slice_rows(X, train_index),
            target[train_index],
            weight,
        )
        prediction = np.asarray(
            estimator.predict(_slice_rows(X, validation_index)), dtype=np.float64
        ).reshape(-1)
        if prediction.shape[0] != validation_index.shape[0]:
            raise ValueError(f"{fold.name} estimator returned the wrong prediction length")
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"{fold.name} prediction contains NaN or infinity")
        all_indices.append(validation_index)
        all_targets.append(target[validation_index])
        all_predictions.append(prediction)
        all_months.append(month[validation_index].astype(np.int16, copy=False))
        all_names.append(np.full(validation_index.size, fold.name, dtype=object))

    row_indices = np.concatenate(all_indices)
    if np.unique(row_indices).size != row_indices.size:
        raise AssertionError("validation folds overlap")
    return OOFResult(
        row_indices=row_indices,
        months=np.concatenate(all_months),
        y_true=np.concatenate(all_targets),
        prediction=np.concatenate(all_predictions),
        fold_names=np.concatenate(all_names),
    )


def run_development_oof(
    X: ArrayLike,
    y: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    estimator_factory: EstimatorFactory,
    *,
    recency_weighting: RecencyWeighting | None = None,
) -> OOFResult:
    """Evaluate one frozen candidate on Dev1--Dev3 only."""

    return _run_fold_set(
        X,
        y,
        months,
        estimator_factory,
        DEVELOPMENT_FOLDS,
        recency_weighting=recency_weighting,
        allow_sealed=False,
    )


def run_sealed_audit(
    X: ArrayLike,
    y: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    frozen_estimator_factory: EstimatorFactory,
    *,
    confirm_frozen: bool = False,
    recency_weighting: RecencyWeighting | None = None,
) -> OOFResult:
    """Evaluate exactly one already-frozen pipeline on months 59--70.

    The explicit confirmation is a procedural tripwire, not a substitute for
    recording the frozen configuration.  Do not pass several factories or use
    the result to reopen feature/hyperparameter selection.
    """

    if not confirm_frozen:
        raise RuntimeError(
            "sealed audit remains closed; freeze the complete pipeline and pass "
            "confirm_frozen=True only for the one final audit"
        )
    return _run_fold_set(
        X,
        y,
        months,
        frozen_estimator_factory,
        (SEALED_AUDIT_FOLD,),
        recency_weighting=recency_weighting,
        allow_sealed=True,
    )


def monthly_diagnostics(
    y_true: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Return exact per-month diagnostics; these do not replace pooled cosine."""

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    month = np.asarray(months).reshape(-1)
    if y.shape != p.shape or y.shape != month.shape:
        raise ValueError("target, prediction, and months must have matching shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(month)):
        raise ValueError("monthly diagnostic input contains NaN or infinity")
    rows: list[dict[str, float | int]] = []
    for current in np.sort(np.unique(month)):
        mask = month == current
        target_norm = float(np.dot(y[mask], y[mask]))
        prediction_norm = float(np.dot(p[mask], p[mask]))
        denominator = np.sqrt(target_norm * prediction_norm)
        score = float(np.dot(y[mask], p[mask]) / denominator) if denominator > 0 else np.nan
        rows.append(
            {
                "month": int(current),
                "n": int(mask.sum()),
                "cosine": score,
                "target_mean": float(np.mean(y[mask])),
                "target_rms": float(np.sqrt(target_norm / mask.sum())),
                "prediction_mean": float(np.mean(p[mask])),
                "prediction_rms": float(np.sqrt(prediction_norm / mask.sum())),
            }
        )
    return pd.DataFrame(rows).sort_values("month", ignore_index=True)


def summarize_monthly_diagnostics(
    diagnostics: pd.DataFrame, pooled_cosine: float
) -> pd.Series:
    """Summarize robustness without elevating raw worst month to a selector."""

    required = {"month", "cosine", "prediction_rms", "prediction_mean"}
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"monthly diagnostics is missing columns: {missing}")
    score = diagnostics["cosine"].dropna().to_numpy(dtype=np.float64)
    if score.size == 0:
        raise ValueError("monthly diagnostics contains no defined cosine")
    bottom_count = max(1, int(ceil(0.25 * score.size)))
    return pd.Series(
        {
            "pooled_cosine": float(pooled_cosine),
            "median_month_cosine": float(np.median(score)),
            "p10_month_cosine": float(np.percentile(score, 10.0)),
            "bottom_quartile_mean_cosine": float(np.mean(np.sort(score)[:bottom_count])),
            "positive_month_share": float(np.mean(score > 0.0)),
            "worst_month_cosine_diagnostic_only": float(np.min(score)),
            "prediction_rms_median": float(np.median(diagnostics["prediction_rms"])),
            "prediction_abs_mean_max": float(np.max(np.abs(diagnostics["prediction_mean"]))),
        }
    )


def rolling_pooled_cosine(
    y_true: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    *,
    windows: Sequence[int] = (3, 6),
) -> pd.DataFrame:
    """Compute pooled cosine over trailing contiguous month windows."""

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    month = np.asarray(months).reshape(-1)
    if y.shape != p.shape or y.shape != month.shape:
        raise ValueError("target, prediction, and months must have matching shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("rolling diagnostic input contains NaN or infinity")
    unique = np.sort(np.unique(month))
    result = pd.DataFrame({"month": unique.astype(int)})
    for window in windows:
        if int(window) != window or window <= 0:
            raise ValueError("rolling windows must be positive integers")
        values = np.full(unique.size, np.nan)
        for end in range(window - 1, unique.size):
            selected = unique[end - window + 1 : end + 1]
            mask = np.isin(month, selected)
            try:
                values[end] = cosine_score(y[mask], p[mask])
            except ValueError:
                values[end] = np.nan
        result[f"cosine_{window}m"] = values
    return result


def compare_development_candidates(
    X: ArrayLike,
    y: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    candidate_factories: Mapping[str, EstimatorFactory],
    *,
    recency_weighting: RecencyWeighting | None = None,
) -> tuple[dict[str, OOFResult], pd.DataFrame]:
    """Compare a small predeclared candidate mapping on development OOF only."""

    if not candidate_factories:
        raise ValueError("candidate_factories must not be empty")
    results: dict[str, OOFResult] = {}
    rows: list[pd.Series] = []
    for name, factory in candidate_factories.items():
        if not name:
            raise ValueError("candidate names must be nonempty")
        result = run_development_oof(
            X,
            y,
            months,
            factory,
            recency_weighting=recency_weighting,
        )
        results[name] = result
        row = result.summary.copy()
        row["candidate"] = name
        rows.append(row)
    leaderboard = pd.DataFrame(rows).set_index("candidate")
    leaderboard = leaderboard.sort_values("pooled_cosine", ascending=False)
    return results, leaderboard


@dataclass(frozen=True)
class BlendSelection:
    """Development-OOF selection for ``w * Ridge + (1-w) * LightGBM``."""

    ridge_weight: float
    pooled_cosine: float
    scores: pd.DataFrame
    ridge_rms: float
    lightgbm_rms: float


def blend_predictions(
    ridge_prediction: Sequence[float] | np.ndarray,
    lightgbm_prediction: Sequence[float] | np.ndarray,
    ridge_weight: float,
    *,
    renormalize: bool = True,
) -> np.ndarray:
    """Blend two vectors after separately normalizing each over all supplied rows."""

    if ridge_weight < 0.0 or ridge_weight > 1.0 or not np.isfinite(ridge_weight):
        raise ValueError("ridge_weight must lie in [0, 1]")
    ridge = rms_normalize(ridge_prediction)
    lightgbm = rms_normalize(lightgbm_prediction)
    if ridge.shape != lightgbm.shape:
        raise ValueError("component prediction shapes do not match")
    combined = ridge_weight * ridge + (1.0 - ridge_weight) * lightgbm
    return rms_normalize(combined) if renormalize else combined


def select_blend_weight(
    y_true: Sequence[float] | np.ndarray,
    ridge_oof: Sequence[float] | np.ndarray,
    lightgbm_oof: Sequence[float] | np.ndarray,
    *,
    grid: Sequence[float] = BLEND_WEIGHT_GRID,
) -> BlendSelection:
    """Select the sole meta-parameter on concatenated development OOF.

    Ties are resolved toward the larger Ridge weight, the lower-complexity
    component.  The returned weight, not the OOF component RMS values, is what
    should be carried to an outer block.  On that block call
    :func:`blend_predictions`, which normalizes each complete outer vector once.
    """

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    ridge = np.asarray(ridge_oof, dtype=np.float64).reshape(-1)
    lightgbm = np.asarray(lightgbm_oof, dtype=np.float64).reshape(-1)
    if y.shape != ridge.shape or y.shape != lightgbm.shape:
        raise ValueError("target and component OOF vectors must have matching shapes")
    ridge_scale, lightgbm_scale = rms(ridge), rms(lightgbm)
    normalized_ridge = rms_normalize(ridge)
    normalized_lightgbm = rms_normalize(lightgbm)
    rows: list[dict[str, float]] = []
    for weight in sorted(set(float(value) for value in grid)):
        if weight < 0.0 or weight > 1.0 or not np.isfinite(weight):
            raise ValueError("all blend weights must lie in [0, 1]")
        prediction = weight * normalized_ridge + (1.0 - weight) * normalized_lightgbm
        try:
            score = cosine_score(y, prediction)
        except ValueError:
            # Exact cancellation can create a zero vector for an interior weight.
            # It is an invalid candidate, not a reason to discard valid endpoints.
            score = -np.inf
        rows.append({"ridge_weight": weight, "pooled_cosine": score})
    if not rows:
        raise ValueError("blend grid must not be empty")
    scores = pd.DataFrame(rows).sort_values(
        ["pooled_cosine", "ridge_weight"], ascending=[False, False], ignore_index=True
    )
    winner = scores.iloc[0]
    return BlendSelection(
        ridge_weight=float(winner["ridge_weight"]),
        pooled_cosine=float(winner["pooled_cosine"]),
        scores=scores.sort_values("ridge_weight", ignore_index=True),
        ridge_rms=ridge_scale,
        lightgbm_rms=lightgbm_scale,
    )


@dataclass(frozen=True)
class BootstrapDifference:
    """Paired moving-month-block uncertainty for cosine(A) - cosine(B)."""

    observed_difference: float
    bootstrap_mean: float
    bootstrap_standard_error: float
    confidence_low: float
    confidence_high: float
    probability_a_better: float
    block_length_months: int
    n_bootstrap: int
    differences: np.ndarray


def paired_month_block_bootstrap(
    y_true: Sequence[float] | np.ndarray,
    prediction_a: Sequence[float] | np.ndarray,
    prediction_b: Sequence[float] | np.ndarray,
    months: Sequence[int] | np.ndarray,
    *,
    block_length_months: int = 3,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    random_state: int = 20260823,
) -> BootstrapDifference:
    """Paired moving-block bootstrap using month-level sufficient statistics.

    Month blocks preserve short-range temporal dependence.  Computing from dot
    products avoids materializing repeated row-level bootstrap samples.
    """

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    a = np.asarray(prediction_a, dtype=np.float64).reshape(-1)
    b = np.asarray(prediction_b, dtype=np.float64).reshape(-1)
    month = np.asarray(months).reshape(-1)
    if y.shape != a.shape or y.shape != b.shape or y.shape != month.shape:
        raise ValueError("target, predictions, and months must have matching shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("bootstrap inputs contain NaN or infinity")
    if int(block_length_months) != block_length_months or block_length_months <= 0:
        raise ValueError("block_length_months must be a positive integer")
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    unique = np.sort(np.unique(month))
    n_months = unique.size
    if unique.size > 1 and not np.all(np.diff(unique) == 1):
        raise ValueError("month-block bootstrap requires consecutive months")
    if block_length_months > n_months:
        raise ValueError("block length exceeds number of observed months")
    position = np.searchsorted(unique, month)
    stats = np.column_stack(
        (
            np.bincount(position, weights=y * a, minlength=n_months),
            np.bincount(position, weights=a * a, minlength=n_months),
            np.bincount(position, weights=y * b, minlength=n_months),
            np.bincount(position, weights=b * b, minlength=n_months),
            np.bincount(position, weights=y * y, minlength=n_months),
        )
    )
    observed = cosine_score(y, a) - cosine_score(y, b)
    rng = np.random.default_rng(random_state)
    max_start = n_months - block_length_months
    n_blocks = int(ceil(n_months / block_length_months))
    differences = np.empty(n_bootstrap, dtype=np.float64)

    for draw in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate(
            [np.arange(start, start + block_length_months) for start in starts]
        )[:n_months]
        multiplicity = np.bincount(sampled, minlength=n_months).astype(np.float64)
        dot_a, norm_a, dot_b, norm_b, norm_y = multiplicity @ stats
        denominator_a = np.sqrt(norm_y * norm_a)
        denominator_b = np.sqrt(norm_y * norm_b)
        differences[draw] = (
            dot_a / denominator_a - dot_b / denominator_b
            if denominator_a > 0.0 and denominator_b > 0.0
            else np.nan
        )
    valid = differences[np.isfinite(differences)]
    if valid.size < 2:
        raise ValueError("bootstrap produced fewer than two defined replicates")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(valid, [tail, 1.0 - tail])
    return BootstrapDifference(
        observed_difference=float(observed),
        bootstrap_mean=float(np.mean(valid)),
        bootstrap_standard_error=float(np.std(valid, ddof=1)),
        confidence_low=float(low),
        confidence_high=float(high),
        probability_a_better=float(np.mean(valid > 0.0)),
        block_length_months=int(block_length_months),
        n_bootstrap=int(valid.size),
        differences=valid,
    )


def _run_synthetic_tests() -> None:
    """Fast dependency-light tests; executed only when running this file directly."""

    # Exact uncentered metric and whole-vector RMS normalization.
    y_small = np.asarray([1.0, -2.0, 3.0])
    assert np.isclose(cosine_score(y_small, 7.0 * y_small), 1.0)
    assert np.isclose(cosine_score(y_small, -y_small), -1.0)
    assert np.isclose(rms(rms_normalize(y_small)), 1.0)

    # Exact fold boundaries, no gap, and a closed sealed-audit tripwire.
    month_grid = np.arange(71)
    train, valid = fold_indices(month_grid, DEVELOPMENT_FOLDS[0])
    assert np.array_equal(train, np.arange(23))
    assert np.array_equal(valid, np.arange(23, 35))
    assert DEVELOPMENT_FOLDS[-1].validation_end == 58
    assert SEALED_AUDIT_FOLD.validation_start == 59

    # Train-fitted robust handling: invalid log values become missing; extreme
    # held-out values are deterministically clipped.
    train_frame = pd.DataFrame(
        {
            "positive": [0.0, 1.0, 2.0, np.nan, 4.0],
            "signed": [-100.0, -1.0, 0.0, 1.0, 100.0],
            "bounded": [-1.0, -0.5, 0.0, 0.5, 1.0],
        }
    )
    processor = RobustPreprocessor(
        feature_kinds={
            "positive": "log1p_nonnegative",
            "signed": "asinh_signed",
            "bounded": "bounded",
        },
        clip=4.0,
    ).fit(train_frame)
    held_out = pd.DataFrame(
        {"positive": [-1.0, 1e30], "signed": [np.inf, -1e30], "bounded": [8.0, -8.0]}
    )
    transformed = processor.transform(held_out)
    assert np.all(np.isfinite(transformed))
    assert np.max(np.abs(transformed[:, :3])) <= 4.0
    assert transformed.shape[1] == 4  # only positive was missing in training

    # Chronological OOF with a fresh pipeline per fold.
    rng = np.random.default_rng(7)
    months = np.repeat(np.arange(71), 12)
    f1 = rng.normal(size=months.size)
    f2 = rng.normal(size=months.size)
    matrix = pd.DataFrame({"f1": f1, "f2": f2})
    target = 1.5 * f1 - 0.4 * f2 + rng.normal(scale=0.03, size=months.size)
    result = run_development_oof(
        matrix,
        target,
        months,
        lambda: make_ridge_estimator(alpha=1.0),
    )
    assert result.row_indices.size == 36 * 12
    assert result.pooled_cosine > 0.99
    assert result.monthly.shape[0] == 36
    assert rolling_pooled_cosine(
        result.y_true, result.prediction, result.months
    )["cosine_6m"].notna().sum() == 31

    try:
        run_sealed_audit(
            matrix,
            target,
            months,
            lambda: make_ridge_estimator(alpha=1.0),
        )
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("sealed audit opened without explicit confirmation")

    # Nonnegative blend should choose the near-perfect Ridge component.
    ridge_pred = result.y_true + rng.normal(scale=0.01, size=result.y_true.size)
    gbdt_pred = rng.normal(size=result.y_true.size)
    blend = select_blend_weight(result.y_true, ridge_pred, gbdt_pred)
    assert blend.ridge_weight == 1.0
    final_blend = blend_predictions(ridge_pred, gbdt_pred, blend.ridge_weight)
    assert np.isclose(rms(final_blend), 1.0)

    views = assemble_prediction_views(np.column_stack((ridge_pred, 2.0 * ridge_pred)))
    assert np.isclose(rms(views), 1.0)
    assert cosine_score(ridge_pred, views) > 0.999999

    weighting = RecencyWeighting(half_life_months=12.0)
    weights = weighting.weights(np.arange(23), train_end=22)
    assert np.isclose(np.mean(weights), 1.0) and weights[-1] > weights[0]

    bootstrap = paired_month_block_bootstrap(
        result.y_true,
        ridge_pred,
        gbdt_pred,
        result.months,
        n_bootstrap=100,
        block_length_months=3,
        random_state=9,
    )
    assert bootstrap.observed_difference > 0.0
    assert bootstrap.probability_a_better > 0.99


if __name__ == "__main__":
    _run_synthetic_tests()
    print("analysis/modeling.py synthetic tests passed")

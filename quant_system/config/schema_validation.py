"""
Pydantic schemas for high-impact config files.

Goal:
    Fail fast on malformed settings before long-running training/tuning jobs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _TaskHorizonMixin(BaseModel):
    model_config = ConfigDict(extra="allow")

    horizon_bars: Optional[int] = Field(default=None, ge=1)
    min_horizon: Optional[int] = Field(default=None, ge=1)
    max_horizon: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_horizon_bounds(self) -> "_TaskHorizonMixin":
        if self.min_horizon is not None and self.max_horizon is not None and self.min_horizon > self.max_horizon:
            raise ValueError("min_horizon must be <= max_horizon")
        return self


class LiqFlowLabelConfig(_TaskHorizonMixin):
    displacement_min_body_pct: float = Field(ge=0.0, le=1.0)
    volume_z_threshold: float
    retrace_min_ratio: float = Field(ge=0.0)
    continuation_min_R: float = Field(gt=0.0)


class BosContLabelConfig(_TaskHorizonMixin):
    min_R: float = Field(gt=0.0)
    bos_min_displacement_body_pct: float = Field(ge=0.0, le=1.0)
    bos_atr_buffer_mult: float = Field(ge=0.0)


class MomoLabelConfig(_TaskHorizonMixin):
    noise_band_sigma: float = Field(gt=0.0)
    return_threshold_sigma: float = Field(gt=0.0)


class Flow1HLabelConfig(_TaskHorizonMixin):
    min_R: float = Field(gt=0.0)
    stop_R: float = Field(gt=0.0)
    min_displacement_body_pct: float = Field(ge=0.0, le=1.0)
    min_volume_z: float
    max_flow_age_bars: int = Field(ge=0)


class EopLabelConfig(_TaskHorizonMixin):
    Aplus_min_conf: float = Field(ge=0.0, le=1.0)
    Aplus_min_evr: float = Field(gt=0.0)
    Aplus_min_medianR: float = Field(gt=0.0)
    hazard_cap: float = Field(ge=0.0)


class EdpLabelConfig(_TaskHorizonMixin):
    drawdown_R_threshold: float


class HazardLabelConfig(_TaskHorizonMixin):
    event_R_threshold: float


class LabelTasksConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    liq_flow: LiqFlowLabelConfig
    bos_cont: BosContLabelConfig
    momo: MomoLabelConfig
    flow_1h: Flow1HLabelConfig
    eop: EopLabelConfig
    edp: EdpLabelConfig
    hazard: HazardLabelConfig


class LabelsTuningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    cv_splits: Optional[int] = Field(default=None, ge=2)
    embargo_bars: Optional[int] = Field(default=None, ge=0)
    default_hpo_trials: Optional[int] = Field(default=None, ge=0)


class LabelsFileConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    labels: LabelTasksConfig
    tuning: Optional[LabelsTuningConfig] = None


class ModelEntryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    algorithm: Optional[str] = None
    calibrator: Optional[str] = None
    hpo_trials: Optional[int] = Field(default=None, ge=0)
    cv_splits: Optional[int] = Field(default=None, ge=2)
    horizon_bars: Optional[int] = Field(default=None, ge=1)

    @field_validator("calibrator")
    @classmethod
    def _validate_calibrator(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"auto", "platt", "isotonic", "none", "empirical"}
        vv = str(v).lower().strip()
        if vv not in allowed:
            raise ValueError(f"unsupported calibrator '{v}'")
        return vv


class FeatureSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    max_missing_ratio: float = Field(ge=0.0, le=1.0)
    drop_exact_duplicates: bool = True
    correlation_threshold: float = Field(gt=0.0, lt=1.0)
    min_features: int = Field(ge=1)
    use_mutual_info: bool = True
    mutual_info_top_k: int = Field(ge=1)
    mutual_info_random_state: int = 42


class ThresholdTuningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    holdout_frac: float = Field(default=0.15, gt=0.0, lt=0.5)
    min_rows: int = Field(default=4096, ge=64)
    min_holdout_rows: int = Field(default=512, ge=32)
    metric: str = "f1"
    min_precision: float = Field(default=0.10, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.01, ge=0.0, le=1.0)
    default_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_candidates: int = Field(default=256, ge=8)

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        vv = str(v).lower().strip()
        if vv not in {"f1", "precision", "recall"}:
            raise ValueError("threshold metric must be one of: f1, precision, recall")
        return vv


class TrainingPreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_imputer: str = "median"
    cat_imputer: str = "most_frequent"
    scaler: str = "standard"
    scale_for_tree_models: bool = False
    outlier_clip: bool = True
    clip_quantiles: list[float] = Field(default_factory=lambda: [0.005, 0.995], min_length=2, max_length=2)
    feature_selection: Optional[FeatureSelectionConfig] = None
    threshold_tuning: Optional[ThresholdTuningConfig] = None

    @field_validator("scaler")
    @classmethod
    def _validate_scaler(cls, v: str) -> str:
        vv = str(v).lower().strip()
        if vv not in {"standard", "robust", "none"}:
            raise ValueError("scaler must be one of: standard, robust, none")
        return vv

    @field_validator("clip_quantiles")
    @classmethod
    def _validate_clip_quantiles(cls, v: list[float]) -> list[float]:
        lo, hi = float(v[0]), float(v[1])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1")
        return [lo, hi]


class TcnDefaultConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    hpo_trials: int = Field(ge=1)
    cv_splits: int = Field(ge=2)
    hpo_adaptive_min_completed_trials: Optional[int] = Field(default=None, ge=1)
    hpo_adaptive_no_improve_trials: Optional[int] = Field(default=None, ge=1)
    hpo_adaptive_min_delta: Optional[float] = Field(default=None, ge=0.0)


class TcnTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default: TcnDefaultConfig
    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class InferencePreferenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    routing_mode: str = "tree"
    challenger_mode: str = "tcn"
    allow_hybrid_explicit: bool = False

    @field_validator("routing_mode", "challenger_mode")
    @classmethod
    def _validate_route_mode(cls, v: str) -> str:
        vv = str(v).lower().strip()
        if vv not in {"tree", "tcn", "hybrid_explicit"}:
            raise ValueError("routing mode must be one of: tree, tcn, hybrid_explicit")
        return vv


class ModelsFileConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    models: Dict[str, ModelEntryConfig]
    training_preprocessing: Optional[TrainingPreprocessingConfig] = None
    tcn_training: Optional[TcnTrainingConfig] = None
    inference_preference: Optional[InferencePreferenceConfig] = None

    @model_validator(mode="after")
    def _validate_required_specialists(self) -> "ModelsFileConfig":
        required = {"liq_flow", "bos_cont", "momo", "flow_1h", "eop", "edp"}
        missing = sorted(required - set(self.models.keys()))
        if missing:
            raise ValueError(f"models section missing required specialists: {missing}")
        return self


def validate_known_config_file(filename: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate high-impact config files and return payload unchanged if valid.
    """
    if not isinstance(payload, dict):
        return payload

    name = str(filename).lower().strip()
    try:
        if name == "models.yaml":
            ModelsFileConfig.model_validate(payload)
        elif name == "labels.yaml":
            LabelsFileConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{filename} schema validation failed:\n{exc}") from exc
    return payload

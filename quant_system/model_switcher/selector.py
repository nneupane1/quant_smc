"""
selector.py
Regime-Conditioned Model Switching Engine.

Chooses ONE model version to run based on:
 • HMM regime probabilities
 • volatility percentile
 • liquidity condition
 • session
 • performance ancestry (historical EVR, precision, DD)
 • disagreement level (if provided)

This provides exclusive selection (not ensemble fusion).
"""

from quant_system.utils.logger import get_logger

LOG = get_logger("model_selector")


class ModelSelector:
    """
    Main switching controller.

    Input:
        model_versions : {model_id: predictor_instance}
        config         : model switching config (from YAML)
        regime_engine  : provides HMM regime states
        perf_memory    : stores per-model historical performance

    Output:
        pick(model_features) -> chosen model_id
    """

    def __init__(self, model_versions, config, regime_engine, perf_memory, registry=None):
        self.models = model_versions
        self.config = config or {}
        self.regime_engine = regime_engine
        self.perf = perf_memory
        self.registry = registry

        self.rules = None
        if self.config.get("rules"):
            from quant_system.model_switcher.regime_rules import RegimeRules
            self.rules = RegimeRules(self.config["rules"])

        LOG.info("[ModelSelector] initialized with models: "
                 f"{list(self.models.keys())}")

    # ------------------------------------------------------------------
    def pick(self, features: dict, disagreement: float = None) -> str:
        """
        Returns model_id
        """

        regime = self._get_regime_state(features)
        vol = features.get("volatility_pctile", features.get("vol_pct", 50))
        session = (
            features.get("session_tag")
            or features.get("session_name")
            or features.get("session")
            or "unknown"
        )
        available = list(self.models.keys())

        # 1) Apply explicit regime rules from YAML
        if self.rules:
            candidate = self.rules.apply(regime, vol, session, available_models=available)
            if candidate in self.models:
                LOG.info(f"[ModelSelector] rule → {candidate}")
                return candidate

        # 2) Performance-aware fallback switching
        ranked = self.perf.rank_models(available_models=available) if self.perf else []
        if not ranked:
            active = self._active_model()
            if active in self.models:
                LOG.info(f"[ModelSelector] no perf data; using active model {active}")
                return active
            # no memory; default to first available model
            fallback = available[0]
            LOG.warning(f"[ModelSelector] no perf data; defaulting to {fallback}")
            return fallback

        best = ranked[0]["model_id"]

        # 3) Disagreement safety override
        if disagreement is not None and disagreement > self.config.get("max_disagreement", 0.25):
            safe_model = self._find_safest_model(ranked)
            LOG.warning(f"[ModelSelector] disagreement override → {safe_model}")
            return safe_model

        LOG.info(f"[ModelSelector] selected {best}")
        return best

    def _get_regime_state(self, features: dict):
        if self.regime_engine and hasattr(self.regime_engine, "get_regime_state"):
            regime = self.regime_engine.get_regime_state()
            if isinstance(regime, dict) and regime:
                return regime
        return {
            k: v
            for k, v in features.items()
            if isinstance(v, (int, float)) and (str(k).startswith("p_regime_") or str(k).startswith("regime_"))
        }

    def _active_model(self):
        if self.registry and hasattr(self.registry, "get_active_model"):
            active = self.registry.get_active_model()
            if isinstance(active, dict):
                return active.get("model_id")
        return None

    # ------------------------------------------------------------------
    def _find_safest_model(self, ranked_list):
        """
        Selects model with lowest DD and CVaR.
        ranked_list: [{model_id, evr, precision, dd, cvar}, ...]
        """
        return sorted(
            ranked_list,
            key=lambda x: (x["dd"], x["cvar"])
        )[0]["model_id"]

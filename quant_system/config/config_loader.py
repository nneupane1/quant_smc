"""
Unified YAML configuration loader with layered merging and .env support.
Supports recursive YAML discovery under the config directory.
"""

import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv
from copy import deepcopy

from quant_system.utils.logger import log
from quant_system.config.schema_validation import validate_known_config_file


class ConfigLoader:
    """
    Loads layered YAML configs from conf/ directory.
    Supports:
        - base.yaml
        - module-level yaml files (recursively)
        - env overrides (DEV/TEST/LIVE)
        - regime/session overrides
        - .env injection
    """

    def __init__(self, conf_dir: str, env: str = None):
        self.conf_dir = conf_dir
        self.env = env or os.getenv("QS_ENV", "DEV").upper()
        self.verbose_file_logs = os.getenv("QS_VERBOSE_CONFIG_LOGS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._reset_load_trace()
        self._load_env_files()
        if self.verbose_file_logs:
            log(f"ConfigLoader initialized. env={self.env}")

    def _reset_load_trace(self):
        self._load_trace = {
            "base_loaded": False,
            "module_files": 0,
            "env_override": False,
            "regime_overrides": 0,
            "session_overrides": 0,
            "env_injected": False,
        }

    def _record_load(self, message: str, trace_key: str = None):
        if trace_key == "base_loaded":
            self._load_trace["base_loaded"] = True
        elif trace_key == "env_override":
            self._load_trace["env_override"] = True
        elif trace_key == "env_injected":
            self._load_trace["env_injected"] = True
        elif trace_key in {"module_files", "regime_overrides", "session_overrides"}:
            self._load_trace[trace_key] += 1

        if self.verbose_file_logs:
            log(message)

    def _load_env_files(self):
        """
        Load config-local secrets first, then fall back to the ambient .env.
        """
        secrets_path = os.path.join(self.conf_dir, "secrets.env")
        if os.path.exists(secrets_path):
            load_dotenv(secrets_path, override=False)
        load_dotenv(override=False)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        filename = os.path.basename(path)
        data = validate_known_config_file(filename, data)
        return data

    def _merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge: override keys replace base keys recursively.
        """
        out = deepcopy(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = self._merge(out[k], v)
            else:
                out[k] = v
        return out

    def _find_file(self, filename: str) -> str:
        """
        Search recursively for a filename under conf_dir.
        """
        for root, dirs, files in os.walk(self.conf_dir):
            if filename in files:
                return os.path.join(root, filename)
        return os.path.join(self.conf_dir, filename)

    def _load_base(self) -> Dict[str, Any]:
        path = os.path.join(self.conf_dir, "base.yaml")
        cfg = self._load_yaml(path)
        self._record_load("Loaded base.yaml", "base_loaded")
        return cfg

    # Public helper for modules that need a single YAML file
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific YAML file from the config directory (recursively).
        """
        path = self._find_file(filename)
        return self._load_yaml(path)

    def _load_modules(self) -> Dict[str, Any]:
        """
        Recursively load all yaml files except base and env override and overrides dirs.
        """
        merged = {}
        skip_dirs = {"regime_overrides", "session_overrides"}
        skip_files = {"base.yaml", f"{self.env}.yaml"}
        yaml_paths = []

        for root, dirs, files in os.walk(self.conf_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if fname.endswith(".yaml") and fname not in skip_files:
                    yaml_paths.append(os.path.join(root, fname))

        for path in sorted(yaml_paths):
            mod = self._load_yaml(path)
            merged = self._merge(merged, mod)
            self._record_load(f"Loaded {os.path.relpath(path, self.conf_dir)}", "module_files")
        return merged

    def _load_env_override(self) -> Dict[str, Any]:
        """
        Load DEV.yaml, TEST.yaml, LIVE.yaml if present.
        """
        fname = f"{self.env}.yaml"
        path = self._find_file(fname)
        if os.path.exists(path):
            self._record_load(f"Loaded env override {fname}", "env_override")
            return self._load_yaml(path)
        return {}

    def _load_regime_overrides(self) -> Dict[str, Any]:
        """
        Merge all overrides in conf/regime_overrides/
        """
        d = os.path.join(self.conf_dir, "regime_overrides")
        if not os.path.exists(d):
            return {}
        merged = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".yaml"):
                path = os.path.join(d, f)
                cfg = self._load_yaml(path)
                merged = self._merge(merged, cfg)
                self._record_load(f"Loaded regime override {f}", "regime_overrides")
        return merged

    def _load_session_overrides(self) -> Dict[str, Any]:
        """
        Merge session overrides in conf/session_overrides/
        """
        d = os.path.join(self.conf_dir, "session_overrides")
        if not os.path.exists(d):
            return {}
        merged = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".yaml"):
                path = os.path.join(d, f)
                cfg = self._load_yaml(path)
                merged = self._merge(merged, cfg)
                self._record_load(f"Loaded session override {f}", "session_overrides")
        return merged

    def _inject_env_vars(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace any config entries that reference env vars like:
            api_key: ${KRAKEN_API_KEY}
        """
        def resolve(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                return os.getenv(key)
            return value

        def recurse(d):
            updated = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    updated[k] = recurse(v)
                else:
                    updated[k] = resolve(v)
            return updated

        cfg2 = recurse(cfg)
        self._record_load("Injected .env variables into configuration.", "env_injected")
        return cfg2

    def _normalize_assets(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        assets = cfg.get("assets")
        if not isinstance(assets, dict):
            assets = {}
            cfg["assets"] = assets

        if "default_asset" not in assets and "default_asset" in cfg:
            assets["default_asset"] = cfg["default_asset"]
        if "metadata" not in assets and isinstance(cfg.get("metadata"), dict):
            assets["metadata"] = cfg["metadata"]

        # Preserve legacy aliases while making the nested form canonical.
        if "default_asset" not in cfg and "default_asset" in assets:
            cfg["default_asset"] = assets["default_asset"]
        if "metadata" not in cfg and isinstance(assets.get("metadata"), dict):
            cfg["metadata"] = assets["metadata"]
        return cfg

    def _normalize_execution(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        exec_cfg = cfg.setdefault("execution", {})
        conf_cfg = exec_cfg.setdefault("confluence", {})
        trade_rate = exec_cfg.setdefault("trade_rate", {})
        tiers = exec_cfg.setdefault("tiers", {})
        hazard_cfg = exec_cfg.setdefault("hazard_trailing", {})
        mpc_cfg = exec_cfg.setdefault("mpc", {})
        stops_targets = exec_cfg.setdefault("stops_targets", {})
        evr_cfg = exec_cfg.setdefault("evr", {})

        if "session_weights" in cfg and "session_weights" not in conf_cfg:
            conf_cfg["session_weights"] = cfg["session_weights"]
        if "weights" in cfg and "legacy_component_weights" not in conf_cfg:
            conf_cfg["legacy_component_weights"] = cfg["weights"]

        thresholds = cfg.get("thresholds", {})
        tr = thresholds.get("trade_rate", {})
        if "target_daily_min" not in trade_rate and "target_min" in tr:
            trade_rate["target_daily_min"] = tr["target_min"]
        if "target_daily_max" not in trade_rate and "target_max" in tr:
            trade_rate["target_daily_max"] = tr["target_max"]
        if "precision_floor" not in trade_rate and "precision_floor" in thresholds:
            trade_rate["precision_floor"] = thresholds["precision_floor"]

        legacy_tiers = thresholds.get("tiers", {})
        if "Aplus" not in tiers and "A_plus" in legacy_tiers:
            aplus = legacy_tiers["A_plus"]
            tiers["Aplus"] = {
                "min_confluence": aplus.get("confluence"),
                "min_evr": aplus.get("evr_min"),
                "min_medianR": aplus.get("median_r"),
                "max_hazard": aplus.get("hazard_max"),
                "auto_execute": True,
            }
        if "A" not in tiers and "A" in legacy_tiers:
            a = legacy_tiers["A"]
            tiers["A"] = {
                "min_confluence": a.get("confluence"),
                "min_evr": a.get("evr_min"),
                "min_medianR": a.get("median_r"),
                "max_hazard": a.get("hazard_max"),
                "auto_execute": False,
            }

        policy_thr = cfg.get("policy", {}).get("thresholds", {})
        if policy_thr and "thresholds" not in hazard_cfg:
            hazard_cfg["thresholds"] = policy_thr

        legacy_mpc = cfg.get("mpc", {})
        if legacy_mpc and "cvar_target" not in mpc_cfg and "cvar_cap" in legacy_mpc:
            mpc_cfg["cvar_target"] = legacy_mpc["cvar_cap"]
        if legacy_mpc and "risk_modes" not in mpc_cfg and "allowed_risk_modes" in legacy_mpc:
            modes = legacy_mpc["allowed_risk_modes"]
            if isinstance(modes, list) and len(modes) >= 3:
                mpc_cfg["risk_modes"] = {"low": modes[0] / 100.0, "medium": modes[1] / 100.0, "high": modes[2] / 100.0}
        if legacy_mpc and "lock_fraction_bounds" not in mpc_cfg and "lock_grid" in legacy_mpc:
            grid = legacy_mpc["lock_grid"]
            if isinstance(grid, list) and grid:
                mpc_cfg["lock_fraction_bounds"] = [min(grid), max(grid)]
        if legacy_mpc and "hedge_ratio_bounds" not in mpc_cfg and "hedge_ratio_grid" in legacy_mpc:
            grid = legacy_mpc["hedge_ratio_grid"]
            if isinstance(grid, list) and grid:
                mpc_cfg["hedge_ratio_bounds"] = [min(grid), max(grid)]

        legacy_stops = cfg.get("stops", {})
        if "stop_atr_mult" not in stops_targets and "atr_multiplier" in legacy_stops:
            stops_targets["stop_atr_mult"] = legacy_stops["atr_multiplier"]
        legacy_targets = cfg.get("targets", {})
        if "target_order" not in stops_targets and legacy_targets:
            order = []
            if legacy_targets.get("swing_levels"):
                order.append("swing")
            if legacy_targets.get("fvg_targets"):
                order.append("fvg")
            if legacy_targets.get("liquidity_levels"):
                order.append("liquidity")
            if order:
                stops_targets["target_order"] = order
        legacy_evr = cfg.get("evr", {})
        cost_bps = legacy_evr.get("cost_bps", {})
        if cost_bps:
            evr_cfg.setdefault("fee_bps", {"taker": cost_bps.get("taker", 4), "maker": cost_bps.get("maker", 2)})
            evr_cfg.setdefault(
                "slippage_bps",
                {
                    "market": cost_bps.get("slippage_market_bps", 3),
                    "limit_hit": cost_bps.get("slippage_limit_bps", 1.5),
                },
            )

        if isinstance(cfg.get("risk"), dict):
            exec_cfg.setdefault("legacy_risk", cfg["risk"])
        if isinstance(cfg.get("fees"), dict):
            exec_cfg.setdefault("legacy_fees", cfg["fees"])
        if isinstance(cfg.get("spot"), dict) or isinstance(cfg.get("perp"), dict):
            exec_cfg.setdefault("legacy_routing", {})
            if isinstance(cfg.get("spot"), dict):
                exec_cfg["legacy_routing"]["spot"] = cfg["spot"]
            if isinstance(cfg.get("perp"), dict):
                exec_cfg["legacy_routing"]["perp"] = cfg["perp"]
        return cfg

    def _normalize_features(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        features = cfg.setdefault("features", {})

        if isinstance(cfg.get("ema"), dict):
            features.setdefault("ema", {})
            features["ema"] = self._merge(features["ema"], cfg["ema"])

        legacy_regimes = cfg.get("regimes", {})
        if legacy_regimes:
            features.setdefault("regime", {})
            hdb = legacy_regimes.get("hdbscan", {})
            hmm = legacy_regimes.get("hmm", {})
            if "hdbscan_min_cluster_size" not in features["regime"] and "min_cluster_size" in hdb:
                features["regime"]["hdbscan_min_cluster_size"] = hdb["min_cluster_size"]
            if "hdbscan_min_samples" not in features["regime"] and "min_samples" in hdb:
                features["regime"]["hdbscan_min_samples"] = hdb["min_samples"]
            if "hmm_states" not in features["regime"] and "n_states" in hmm:
                features["regime"]["hmm_states"] = hmm["n_states"]
            if "hmm_covariance" not in features["regime"] and "covariance_type" in hmm:
                features["regime"]["hmm_covariance"] = hmm["covariance_type"]
            features.setdefault("regime_legacy", legacy_regimes)

        smc = features.setdefault("smc", {})
        if isinstance(cfg.get("swings"), dict):
            smc.setdefault("swings", cfg["swings"])
        if isinstance(cfg.get("bos"), dict):
            smc.setdefault("bos", cfg["bos"])
        if isinstance(cfg.get("choch"), dict):
            smc.setdefault("choch", cfg["choch"])
        if isinstance(cfg.get("fvg"), dict):
            smc.setdefault("fvg", cfg["fvg"])
        if isinstance(cfg.get("sweeps"), dict) and "sweep" not in smc:
            smc["sweep"] = cfg["sweeps"]
        if isinstance(cfg.get("order_blocks"), dict) and "zones" not in smc:
            smc["zones"] = cfg["order_blocks"]
        if isinstance(cfg.get("context"), dict) and "context_legacy" not in smc:
            smc["context_legacy"] = cfg["context"]

        paths = cfg.setdefault("paths", {})
        if "model_registry" not in paths and cfg.get("models", {}).get("registry_path"):
            paths["model_registry"] = cfg["models"]["registry_path"]

        tcfg = cfg.setdefault("timeframes", {})
        derived = tcfg.get("derived", [])
        if derived:
            names = {item.get("name") for item in derived if isinstance(item, dict)}
            if "6h" not in names and "5h" in names:
                for item in derived:
                    if isinstance(item, dict) and item.get("name") == "5h":
                        item["name"] = "6h"
                        item["multiple"] = 360
            if "12h" not in names and "10h" in names:
                for item in derived:
                    if isinstance(item, dict) and item.get("name") == "10h":
                        item["name"] = "12h"
                        item["multiple"] = 720
        return cfg

    def _normalize_models(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        models = cfg.setdefault("models", {})
        alias_pairs = {
            "liquidity_flow": "liq_flow",
            "bos_continuation": "bos_cont",
            "micro_momentum": "momo",
            "meta": "meta_model",
        }
        for old_name, new_name in alias_pairs.items():
            if new_name not in models and old_name in models:
                models[new_name] = deepcopy(models[old_name])
            if old_name not in models and new_name in models:
                models[old_name] = deepcopy(models[new_name])
        return cfg

    def _validate(self, cfg: Dict[str, Any]):
        required_sections = ["execution", "features", "labels", "models", "assets"]
        missing = [key for key in required_sections if key not in cfg]
        if missing:
            raise KeyError(f"Missing required config sections: {missing}")

        assets = cfg.get("assets", {})
        if "default_asset" not in assets:
            raise KeyError("assets.default_asset is required in merged config.")
        if "metadata" not in assets or not isinstance(assets["metadata"], dict):
            raise KeyError("assets.metadata is required in merged config.")

        exec_cfg = cfg.get("execution", {})
        required_exec = ["capital", "confluence", "gates", "tiers", "hazard_trailing", "profit_ladder", "mpc", "trade_rate"]
        missing_exec = [key for key in required_exec if key not in exec_cfg]
        if missing_exec:
            raise KeyError(f"Missing execution config keys: {missing_exec}")

        tf = cfg.get("timeframes", {})
        canonical = {tf.get("execution"), tf.get("flow"), tf.get("structure"), tf.get("regime")}
        if {"15m", "1h", "6h", "12h"} - canonical:
            raise KeyError("Canonical timeframe mapping must include 15m/1h/6h/12h.")

    def _normalize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = deepcopy(cfg)
        cfg = self._normalize_assets(cfg)
        cfg = self._normalize_execution(cfg)
        cfg = self._normalize_features(cfg)
        cfg = self._normalize_models(cfg)
        self._validate(cfg)
        return cfg

    def load(self) -> Dict[str, Any]:
        """
        Full resolved config.
        Order of precedence:
            base.yaml
            module yamls (recursive)
            env override
            regime overrides
            session overrides
            .env injection
        """
        self._reset_load_trace()
        cfg = self._load_base()
        cfg = self._merge(cfg, self._load_modules())
        cfg = self._merge(cfg, self._load_env_override())
        cfg = self._merge(cfg, self._load_regime_overrides())
        cfg = self._merge(cfg, self._load_session_overrides())
        cfg = self._inject_env_vars(cfg)
        cfg = self._normalize(cfg)
        if self.verbose_file_logs:
            log("Final unified config loaded.")
        else:
            log(
                "Config ready "
                f"env={self.env} "
                f"base={'yes' if self._load_trace['base_loaded'] else 'no'} "
                f"modules={self._load_trace['module_files']} "
                f"env_override={'yes' if self._load_trace['env_override'] else 'no'} "
                f"regime_overrides={self._load_trace['regime_overrides']} "
                f"session_overrides={self._load_trace['session_overrides']} "
                f"env_injected={'yes' if self._load_trace['env_injected'] else 'no'}"
            )
        return cfg

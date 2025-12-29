"""
Python bridge for the Rust pipeline (PyO3 module).
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import List, Tuple


def is_available() -> bool:
    try:
        import sofia_rust_pipeline  # noqa: F401
    except ImportError:
        return False
    return True


def init_trades_cache(cfg) -> int:
    if not is_available():
        raise RuntimeError("Rust pipeline module is not available")

    import sofia_rust_pipeline

    config_json = json.dumps(
        serialize_config(cfg),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    result = sofia_rust_pipeline.init_trades_cache(config_json)
    return int(result.get("trade_count", 0))

def build_trade_index(cfg):
    if not is_available():
        raise RuntimeError("Rust pipeline module is not available")

    import sofia_rust_pipeline

    config_json = json.dumps(
        serialize_config(cfg),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    base_tf_seconds = int(cfg.features.timeframes[cfg.base_timeframe_idx])
    return sofia_rust_pipeline.TradeIndex(config_json, base_tf_seconds)


def _serialize_dataclass(dc) -> dict:
    data = asdict(dc)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def serialize_config(cfg) -> dict:
    return {
        "data": _serialize_dataclass(cfg.data),
        "features": _serialize_dataclass(cfg.features),
        "labels": _serialize_dataclass(cfg.labels),
        "model": _serialize_dataclass(cfg.model),
        "base_timeframe_idx": cfg.base_timeframe_idx,
        "seed": cfg.seed,
    }


def _config_cache_key(cfg) -> str:
    payload = json.dumps(
        serialize_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _read_arrow_dataset(path: Path):
    import pyarrow.ipc as ipc

    with path.open("rb") as f:
        reader = ipc.open_file(f)
        table = reader.read_all()
    return table.to_pandas()


def _read_arrow_bytes(data: bytes):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    reader = ipc.open_file(pa.BufferReader(data))
    table = reader.read_all()
    return table.to_pandas()


def build_dataset_from_config(
    cfg,
    cache_dir: str | Path = "rust_cache",
    write_intermediate: bool = False,
    force: bool = False,
    memory_only: bool = True,
) -> Tuple["object", List[str], Path | None]:
    if not is_available():
        raise RuntimeError("Rust pipeline module is not available")

    import sofia_rust_pipeline

    if memory_only:
        config_json = json.dumps(
            serialize_config(cfg),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        try:
            sofia_rust_pipeline.init_trades_cache(config_json)
        except Exception:
            pass
        result = sofia_rust_pipeline.run_pipeline_memory_json(config_json)
        feature_cols = list(result.get("feature_cols", []))
        dataset = _read_arrow_bytes(result.get("dataset_bytes", b""))
        return dataset, feature_cols, None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _config_cache_key(cfg)
    output_dir = cache_dir / cache_key
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.arrow"
    feature_cols_path = output_dir / "feature_cols.json"

    config_path = output_dir / "config.json"
    if force or not config_path.exists():
        config_path.write_text(
            json.dumps(serialize_config(cfg), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    if force or not (dataset_path.exists() and feature_cols_path.exists()):
        try:
            config_json = json.dumps(
                serialize_config(cfg),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            sofia_rust_pipeline.init_trades_cache(config_json)
        except Exception:
            pass
        result = sofia_rust_pipeline.run_pipeline(
            str(config_path),
            str(output_dir),
            write_intermediate,
        )
        feature_cols = list(result.get("feature_cols", []))
        if not feature_cols and feature_cols_path.exists():
            feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    else:
        feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    dataset = _read_arrow_dataset(dataset_path)
    return dataset, feature_cols, dataset_path

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Tuple
from config import TrendFollowerConfig
from trainer import run_training_pipeline
from models import TrendFollowerModels
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from labels import create_training_dataset
from backtest import SimpleBacktester, print_backtest_results, save_backtest_logs
from optimizer import run_optimization, TrendFollowerOptimizer, OptimizerConfig


_TRAIN_CONFIG_FILENAME = "train_config.json"


def _normalize_config(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _normalize_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config(v) for v in obj]
    return obj


def _serialize_config(cfg: TrendFollowerConfig) -> dict:
    return _normalize_config(asdict(cfg))


def _apply_config_section(target, data: dict, path_fields: Optional[set] = None) -> None:
    path_fields = path_fields or set()
    for key, value in data.items():
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(target, key, value)


def _load_train_config(model_dir: Path) -> Optional[TrendFollowerConfig]:
    config_path = Path(model_dir) / _TRAIN_CONFIG_FILENAME
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = TrendFollowerConfig()
    if isinstance(data, dict):
        if "data" in data:
            _apply_config_section(cfg.data, data["data"], {"data_dir"})
        if "features" in data:
            _apply_config_section(cfg.features, data["features"])
        if "labels" in data:
            _apply_config_section(cfg.labels, data["labels"])
        if "model" in data:
            _apply_config_section(cfg.model, data["model"], {"model_dir"})
        if "base_timeframe_idx" in data:
            cfg.base_timeframe_idx = int(data["base_timeframe_idx"])
        if "seed" in data:
            cfg.seed = int(data["seed"])
    return cfg


def _save_train_config(cfg: TrendFollowerConfig, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / _TRAIN_CONFIG_FILENAME
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_config(cfg), f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--backtest-only',
        action='store_true',
        help='Skip training and only run the backtest using existing saved models.',
    )
    mode.add_argument(
        '--train-only',
        action='store_true',
        help='Train models only (skip the backtest).',
    )
    parser.add_argument(
        '--two-pass',
        action='store_true',
        help='Two-pass training: train with validation (early stopping), then retrain on Train+Val with best iterations.',
    )
    parser.add_argument(
        '--stop-loss-atr',
        type=float,
        default=1.0,
        help='Stop loss in ATR units (default: 1.0).',
    )
    parser.add_argument(
        '--stop-padding-pct',
        type=float,
        default=0.0,
        help='Extra stop distance as a fraction of entry price (default: 0.0 = disabled).',
    )
    parser.add_argument(
        '--take-profit-rr',
        type=float,
        default=1.5,
        help='Take profit reward:risk ratio (default: 1.5).',
    )
    parser.add_argument(
        '--min-bounce-prob',
        type=float,
        default=0.48,
        help='Minimum bounce probability gate (default: 0.48).',
    )
    parser.add_argument(
        '--max-bounce-prob',
        type=float,
        default=1.0,
        help='Maximum bounce probability for bucket filtering (default: 1.0 = no max).',
    )
    parser.add_argument(
        '--use-dynamic-rr',
        action='store_true',
        help='Use expected RR from model for dynamic TP sizing instead of fixed take-profit-rr.',
    )
    parser.add_argument(
        '--touch-threshold-atr',
        type=float,
        default=0.3,
        help='EMA touch detection threshold in ATR units (default: 0.3).',
    )
    parser.add_argument(
        '--ema-touch-mode',
        type=str,
        default='base',
        choices=['base', 'multi'],
        help='EMA touch detection mode for backtest (default: base).',
    )
    parser.add_argument(
        '--cooldown-bars-after-stop',
        type=int,
        default=0,
        help='Cooldown after a stop-loss in base bars (default: 0 = disabled).',
    )
    parser.add_argument(
        '--trade-side',
        type=str,
        default='long',
        choices=['long', 'short', 'both'],
        help='Trade direction to allow in backtest (default: long).',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/MONUSDT',
        help="Directory containing the symbol's trade CSV files (default: data/MONUSDT).",
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models_ema9_touch',
        help='Directory to save/load the models.',
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=None,
        help='If set, only use the most recent N days of trades from the dataset.',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=None,
        help='Train split ratio (fraction). Provide at least two ratios.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=None,
        help='Validation split ratio (fraction). Provide at least two ratios.',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=None,
        help='Test split ratio (fraction). Provide at least two ratios.',
    )

    parser.add_argument('--learning-rate', type=float, default=None, help='LightGBM learning rate')
    parser.add_argument('--num-leaves', type=int, default=None, help='LightGBM num leaves')
    parser.add_argument('--n-estimators', type=int, default=None, help='LightGBM n estimators')
    parser.add_argument('--max-depth', type=int, default=None, help='LightGBM max depth')

    parser.add_argument('--feature-fraction', type=float, default=None, help='LightGBM feature fraction')
    parser.add_argument('--lambdaa-ele1', type=float, default=None, help='LightGBM lambda l1')
    parser.add_argument('--lambdaa-ele2', type=float, default=None, help='LightGBM lambda l2')
    parser.add_argument('--min-child-samples', type=int, default=None, help='LightGBM min child samples')

    # Custom optimizer
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run custom parameter optimizer to find best LightGBM params and min_bounce_prob.',
    )
    parser.add_argument(
        '--optimize-trials',
        type=int,
        default=30,
        help='Number of random search trials for optimization (default: 30).',
    )

    # =========================================================================
    # OPTUNA CONFIG TUNER
    # =========================================================================
    parser.add_argument(
        '--optuna-tune',
        action='store_true',
        help='Run Optuna tuning across config parameters to maximize validation accuracy/precision.',
    )
    parser.add_argument(
        '--tune-scope',
        type=str,
        default='full',
        choices=['model', 'features', 'labels', 'full', 'all'],
        help='Which config sections to tune (default: full).',
    )
    parser.add_argument(
        '--tune-lgbm-only',
        action='store_true',
        help='Alias for --tune-scope model (only LightGBM parameters).',
    )
    parser.add_argument(
        '--tune-trials',
        type=int,
        default=200,
        help='Number of Optuna trials (default: 200).',
    )
    parser.add_argument(
        '--tune-timeout-min',
        type=float,
        default=None,
        help='Optional Optuna timeout in minutes (default: no timeout).',
    )
    parser.add_argument(
        '--tune-precision-weight',
        type=float,
        default=0.6,
        help='Weight for entry-model precision in the tuning objective (0-1).',
    )
    parser.add_argument(
        '--tune-trend-weight',
        type=float,
        default=0.0,
        help='Optional weight for trend validation accuracy in the objective (default: 0).',
    )
    parser.add_argument(
        '--tune-min-pullback-samples',
        type=int,
        default=100,
        help='Minimum pullback samples in train set to score a trial (default: 100).',
    )
    parser.add_argument(
        '--tune-min-pullback-val-samples',
        type=int,
        default=20,
        help='Minimum pullback samples in validation set to score a trial (default: 20).',
    )
    parser.add_argument(
        '--tune-seed',
        type=int,
        default=42,
        help='Random seed for Optuna sampler (default: 42).',
    )
    parser.add_argument(
        '--tune-save-results',
        type=str,
        default=None,
        help='Optional path to save tuning summary JSON (default: model-dir/tuning_summary.json).',
    )
    parser.add_argument(
        '--train-from-tuning',
        nargs='?',
        const='__MODEL_DIR__',
        default=None,
        help='Train models from a tuning summary JSON (uses best_config). '
             'If no path is provided, uses model-dir/tuning_summary.json.',
    )
    parser.add_argument(
        '--tune-then-train',
        action='store_true',
        help='After tuning, train and save models using the best config.',
    )
    parser.add_argument(
        '--tune-no-progress',
        action='store_true',
        help='Disable Optuna progress bar.',
    )

    # =========================================================================
    # MODEL IMPROVEMENT FLAGS (v1.2)
    # =========================================================================
    parser.add_argument(
        '--use-noise-filtering',
        action='store_true',
        help='Enable Noise Injection Feature Selection. Trains a quick model with a random '
             'noise column and removes features that rank below noise in importance.',
    )
    parser.add_argument(
        '--use-seed-ensemble',
        action='store_true',
        help='Enable Seed Ensembling (Bagging). Trains N LightGBM models with different '
             'random seeds and averages their predictions for more stable results.',
    )
    parser.add_argument(
        '--n-ensemble-seeds',
        type=int,
        default=5,
        help='Number of seeds for seed ensembling (default: 5). Only used if --use-seed-ensemble is set.',
    )

    # =========================================================================
    # PROBABILITY CALIBRATION FLAGS
    # =========================================================================
    parser.add_argument(
        '--use-calibration',
        action='store_true',
        help='Use calibrated probabilities (Isotonic Regression) instead of raw model output. '
             'Calibrator is trained on validation set to map raw probabilities to true frequencies.',
    )
    parser.add_argument(
        '--use-raw-probabilities',
        action='store_true',
        help='Explicitly use raw (uncalibrated) probabilities. This is the default behavior. '
             'Mutually exclusive with --use-calibration.',
    )

    args = parser.parse_args()

    if args.tune_lgbm_only:
        args.tune_scope = 'model'

    # Validate mutually exclusive calibration flags
    if args.use_calibration and args.use_raw_probabilities:
        raise SystemExit("Cannot use both --use-calibration and --use-raw-probabilities. Choose one.")

    cfg = TrendFollowerConfig()
    if args.learning_rate is not None:
        cfg.model.learning_rate = args.learning_rate
    if args.num_leaves is not None:
        cfg.model.num_leaves = args.num_leaves
    if args.n_estimators is not None:
        cfg.model.n_estimators = args.n_estimators
    if args.max_depth is not None:
        cfg.model.max_depth = args.max_depth

    if args.feature_fraction is not None:
        cfg.model.feature_fraction = args.feature_fraction
    if args.lambdaa_ele1 is not None:
        cfg.model.lambdaa_ele1 = args.lambdaa_ele1
    if args.lambdaa_ele2 is not None:
        cfg.model.lambdaa_ele2 = args.lambdaa_ele2
    if args.min_child_samples is not None:
        cfg.model.min_child_samples = args.min_child_samples
        
    cfg.data.data_dir = Path(args.data_dir)
    cfg.model.model_dir = Path(args.model_dir)
    cfg.data.lookback_days = args.lookback_days
    cfg.base_timeframe_idx = 1  # 5m
    cfg.features.ema_periods = [9]
    cfg.labels.pullback_ema = 9
    # Align labels with touch logic (intrabar touch within 0.02 ATR)
    cfg.labels.pullback_threshold = 0.02
    cfg.sample_rate = 1.0
    cfg.enable_diagnostics = False

    # Optional: override train/val/test ratios
    ratio_override: Optional[Tuple[float, float, float]] = None
    ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio,
    }
    provided = {k: v for k, v in ratios.items() if v is not None}
    if provided:
        if len(provided) < 2:
            # Special-case: allow "train on the whole set" via --train-ratio 1.0
            if args.train_ratio is not None and float(args.train_ratio) == 1.0:
                train_ratio = 1.0
                val_ratio = 0.0
                test_ratio = 0.0
            else:
                raise SystemExit("Please provide at least two of --train-ratio/--val-ratio/--test-ratio.")
        else:
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            test_ratio = args.test_ratio

            if train_ratio is None:
                train_ratio = 1.0 - float(val_ratio) - float(test_ratio)
            elif val_ratio is None:
                val_ratio = 1.0 - float(train_ratio) - float(test_ratio)
            elif test_ratio is None:
                test_ratio = 1.0 - float(train_ratio) - float(val_ratio)

        for name, value in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
            if value is None:
                raise SystemExit("Could not infer missing ratio; provide at least two ratios.")
            if not (0.0 <= float(value) <= 1.0):
                raise SystemExit(f"Invalid {name} ratio: {value}. Ratios must be between 0 and 1 (inclusive).")

        total = float(train_ratio) + float(val_ratio) + float(test_ratio)
        if abs(total - 1.0) > 1e-6:
            raise SystemExit(f"Ratios must sum to 1.0. Got total={total:.6f}.")

        if not args.backtest_only and float(train_ratio) <= 0.0:
            raise SystemExit("Train ratio must be > 0 when training is enabled.")

        cfg.model.train_ratio = float(train_ratio)
        cfg.model.val_ratio = float(val_ratio)
        cfg.model.test_ratio = float(test_ratio)
        ratio_override = (cfg.model.train_ratio, cfg.model.val_ratio, cfg.model.test_ratio)

    if args.two_pass and not args.backtest_only and float(cfg.model.val_ratio) <= 0.0:
        raise SystemExit("--two-pass requires a non-zero validation split (set --val-ratio > 0).")
    if args.optuna_tune and float(cfg.model.val_ratio) <= 0.0:
        raise SystemExit("--optuna-tune requires a non-zero validation split (set --val-ratio > 0).")

    if args.optimize and args.optuna_tune:
        raise SystemExit("Choose only one: --optimize or --optuna-tune.")
    if args.optuna_tune and args.backtest_only:
        raise SystemExit("--optuna-tune requires training data (remove --backtest-only).")

    if not (0.0 <= float(args.tune_precision_weight) <= 1.0):
        raise SystemExit("--tune-precision-weight must be between 0 and 1.")
    if float(args.tune_trend_weight) < 0.0:
        raise SystemExit("--tune-trend-weight must be >= 0.")

    if args.backtest_only:
        loaded_cfg = _load_train_config(Path(args.model_dir))
        if loaded_cfg is not None:
            cfg = loaded_cfg
            cfg.data.data_dir = Path(args.data_dir)
            cfg.model.model_dir = Path(args.model_dir)
            cfg.data.lookback_days = args.lookback_days
            if ratio_override is not None:
                cfg.model.train_ratio, cfg.model.val_ratio, cfg.model.test_ratio = ratio_override
            print(f"Loaded training config from {Path(args.model_dir) / _TRAIN_CONFIG_FILENAME}")
        print("Backtest-only: skipping training. Using models from", cfg.model.model_dir)
    elif args.optimize:
        # Run custom parameter optimization
        print("Running parameter optimization...")
        print("Loading data for optimization...")

        trades = load_trades(cfg.data, verbose=True)
        trades = preprocess_trades(trades, cfg.data)
        bars = create_multi_timeframe_bars(trades, cfg.features.timeframes, cfg.features.timeframe_names, cfg.data)
        base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]

        print("Calculating features...")
        featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)

        print("Labeling...")
        labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

        # Split into train/val/test
        train_end = int(len(labeled) * cfg.model.train_ratio)
        val_end = int(len(labeled) * (cfg.model.train_ratio + cfg.model.val_ratio))

        train_df = labeled.iloc[:train_end]
        val_df = labeled.iloc[train_end:val_end]

        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        # Get pullback samples for entry model optimization
        pullback_mask_train = ~train_df['pullback_success'].isna()
        pullback_mask_val = ~val_df['pullback_success'].isna()

        X_train = train_df[feature_cols].loc[pullback_mask_train]
        y_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int).values

        X_val = val_df[feature_cols].loc[pullback_mask_val]
        y_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int).values

        print(f"Pullback samples - Train: {len(X_train)}, Val: {len(X_val)}")

        # Run optimization
        opt_config = OptimizerConfig(
            n_random_trials=args.optimize_trials,
            target_rr=float(args.take_profit_rr),
            min_trades_required=15,
        )

        optimizer = TrendFollowerOptimizer(X_train, y_train, X_val, y_val, config=opt_config)
        opt_results = optimizer.optimize(verbose=True)

        # Print recommended command
        if opt_results['best_params']:
            best_params = optimizer.get_best_config_params()
            print("\n" + "=" * 80)
            print("RECOMMENDED TRAINING COMMAND")
            print("=" * 80)
            cmd = (
                f"python train.py --data-dir {args.data_dir} --model-dir {args.model_dir} "
                f"--two-pass --trade-side {args.trade_side} "
                f"--min-bounce-prob {opt_results['best_min_bounce_prob']:.2f} "
                f"--learning-rate {best_params['learning_rate']:.4f} "
                f"--max-depth {best_params['max_depth']} "
                f"--num-leaves {best_params['num_leaves']} "
                f"--n-estimators {best_params['n_estimators']} "
                f"--min-child-samples {best_params['min_child_samples']} "
                f"--feature-fraction {best_params['feature_fraction']:.2f} "
                f"--lambdaa-ele1 {best_params['lambdaa_ele1']:.4f} "
                f"--lambdaa-ele2 {best_params['lambdaa_ele2']:.4f}"
            )
            print(cmd)
            print("=" * 80)

        return  # Don't continue to backtest after optimization
    elif args.train_from_tuning:
        tuning_path_value = args.train_from_tuning
        if tuning_path_value == '__MODEL_DIR__':
            tuning_path = Path(args.model_dir) / "tuning_summary.json"
        else:
            tuning_path = Path(tuning_path_value)
        if not tuning_path.exists():
            raise SystemExit(f"Tuning summary not found: {tuning_path}")
        with open(tuning_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best_config = summary.get("best_config")
        if not isinstance(best_config, dict):
            raise SystemExit("Tuning summary does not contain best_config.")

        tuned_cfg = TrendFollowerConfig()
        _apply_config_section(tuned_cfg.data, best_config.get("data", {}), {"data_dir"})
        _apply_config_section(tuned_cfg.features, best_config.get("features", {}))
        _apply_config_section(tuned_cfg.labels, best_config.get("labels", {}))
        _apply_config_section(tuned_cfg.model, best_config.get("model", {}), {"model_dir"})
        if "base_timeframe_idx" in best_config:
            tuned_cfg.base_timeframe_idx = int(best_config["base_timeframe_idx"])
        if "seed" in best_config:
            tuned_cfg.seed = int(best_config["seed"])

        tuned_cfg.data.data_dir = Path(args.data_dir)
        tuned_cfg.model.model_dir = Path(args.model_dir)
        tuned_cfg.data.lookback_days = args.lookback_days

        if ratio_override is not None:
            tuned_cfg.model.train_ratio, tuned_cfg.model.val_ratio, tuned_cfg.model.test_ratio = ratio_override

        if float(tuned_cfg.model.val_ratio) <= 0.0:
            raise SystemExit("--train-from-tuning requires a non-zero validation split (set --val-ratio > 0).")

        print(f"Training models from tuning summary: {tuning_path}")
        run_training_pipeline(
            tuned_cfg,
            enable_diagnostics=False,
            two_pass=bool(args.two_pass),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
        )
        _save_train_config(tuned_cfg, tuned_cfg.model.model_dir)
        print("Training done. Saving models to", tuned_cfg.model.model_dir)
        return
    elif args.optuna_tune:
        print("Running Optuna config tuning...")
        print("Loading data for tuning...")

        trades = load_trades(cfg.data, verbose=True)
        trades = preprocess_trades(trades, cfg.data)

        try:
            from config_tuner import run_config_tuning, serialize_config
        except ImportError as exc:
            raise SystemExit(str(exc))

        tune_results = run_config_tuning(
            trades=trades,
            base_config=cfg,
            tune_scope=str(args.tune_scope),
            n_trials=int(args.tune_trials),
            timeout_minutes=args.tune_timeout_min,
            precision_weight=float(args.tune_precision_weight),
            trend_weight=float(args.tune_trend_weight),
            min_pullback_samples=int(args.tune_min_pullback_samples),
            min_pullback_val_samples=int(args.tune_min_pullback_val_samples),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
            seed=int(args.tune_seed),
            show_progress=not bool(args.tune_no_progress),
        )

        best_cfg = tune_results.best_config
        best_cfg.model.model_dir = Path(args.model_dir)
        best_cfg.data.data_dir = Path(args.data_dir)

        print("\n" + "=" * 80)
        print("OPTUNA TUNING SUMMARY")
        print("=" * 80)
        print(f"Best Score:        {tune_results.best_score:.6f}")
        print(f"Trials Completed:  {tune_results.trials_completed}")
        print(f"Elapsed Seconds:   {tune_results.elapsed_seconds:.1f}")
        print("Best Metrics:")
        for key, value in tune_results.best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)

        summary_path = Path(args.tune_save_results) if args.tune_save_results else Path(args.model_dir) / "tuning_summary.json"
        summary = {
            'best_score': tune_results.best_score,
            'best_params': tune_results.best_params,
            'best_metrics': tune_results.best_metrics,
            'best_config': serialize_config(best_cfg),
            'trials_completed': tune_results.trials_completed,
            'elapsed_seconds': tune_results.elapsed_seconds,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Tuning summary saved to {summary_path}")

        if args.tune_then_train:
            print("\nTraining models using best config from tuning...")
            run_training_pipeline(
                best_cfg,
                enable_diagnostics=False,
                two_pass=bool(args.two_pass),
                use_noise_filtering=bool(args.use_noise_filtering),
                use_seed_ensemble=bool(args.use_seed_ensemble),
                n_ensemble_seeds=int(args.n_ensemble_seeds),
            )
            _save_train_config(best_cfg, best_cfg.model.model_dir)
            print("Training done. Saving models to", best_cfg.model.model_dir)
        else:
            print("Tuning complete. Use --tune-then-train to train with the best config.")
        return
    else:
        print("Training models...")
        run_training_pipeline(
            cfg,
            enable_diagnostics=False,
            two_pass=bool(args.two_pass),
            use_noise_filtering=bool(args.use_noise_filtering),
            use_seed_ensemble=bool(args.use_seed_ensemble),
            n_ensemble_seeds=int(args.n_ensemble_seeds),
        )
        _save_train_config(cfg, cfg.model.model_dir)
        print("Training done. Saving models to", cfg.model.model_dir)

    if args.train_only:
        print("Train-only: skipping backtest.")
        return

    print("Loading models for backtest...")
    models = TrendFollowerModels(cfg.model)
    models.load_all(cfg.model.model_dir)

    print("Loading data...")
    trades = load_trades(cfg.data, verbose=False)
    trades = preprocess_trades(trades, cfg.data)

    print("Creating bars...")
    bars = create_multi_timeframe_bars(trades, cfg.features.timeframes, cfg.features.timeframe_names, cfg.data)
    base_tf = cfg.features.timeframe_names[cfg.base_timeframe_idx]

    print("Calculating features...")
    featured = calculate_multi_timeframe_features(bars, base_tf, cfg.features)

    print("Labeling...")
    labeled, feature_cols = create_training_dataset(featured, cfg.labels, cfg.features, base_tf)

    start = int(len(labeled) * (cfg.model.train_ratio + cfg.model.val_ratio))
    test = labeled.iloc[start:]
    if len(test) == 0:
        raise SystemExit(
            "Test split is empty (test_ratio=0). Use --train-only to just train, or set a non-zero --test-ratio."
        )
    print(f"Test set size: {len(test)} bars from index {start}")

    bt = SimpleBacktester(
        models,
        cfg,
        min_bounce_prob=float(args.min_bounce_prob),
        max_bounce_prob=float(args.max_bounce_prob),
        min_quality=getattr(cfg, 'min_quality', 'B'),
        stop_loss_atr=float(args.stop_loss_atr),
        stop_padding_pct=float(args.stop_padding_pct),
        take_profit_rr=float(args.take_profit_rr),
        cooldown_bars_after_stop=int(args.cooldown_bars_after_stop),
        trade_side=str(args.trade_side),
        use_dynamic_rr=bool(args.use_dynamic_rr),
        use_ema_touch_entry=True,  # Always use touch-based entry
        ema_touch_mode=str(args.ema_touch_mode),
        touch_threshold_atr=float(args.touch_threshold_atr),
        raw_trades=trades,  # Pass raw trades for precise TP/SL detection
        use_calibration=bool(args.use_calibration),  # Use calibrated probabilities
    )
    res = bt.run(test, feature_cols)
    print_backtest_results(res)

    stop_stats = {}
    if res.trades:
        stops = []
        for t in res.trades:
            if t.stop_loss is not None and t.entry_price:
                dist_pct = abs(t.entry_price - t.stop_loss) / t.entry_price * 100
                stops.append(dist_pct)
        if stops:
            import pandas as pd
            s = pd.Series(stops)
            stop_stats = {
                'stop_distance_pct_mean': float(s.mean()),
                'stop_distance_pct_median': float(s.median()),
                'stop_distance_pct_min': float(s.min()),
                'stop_distance_pct_max': float(s.max()),
            }
            print(
                "Stop distance pct (entry to stop) -> "
                f"mean {stop_stats['stop_distance_pct_mean']:.3f}%, "
                f"median {stop_stats['stop_distance_pct_median']:.3f}%, "
                f"min {stop_stats['stop_distance_pct_min']:.3f}%, "
                f"max {stop_stats['stop_distance_pct_max']:.3f}%"
            )
        else:
            print("Stop distance pct: no data")

    log_params = {
        'backtest_only': bool(args.backtest_only),
        'min_bounce_prob': float(args.min_bounce_prob),
        'max_bounce_prob': float(args.max_bounce_prob),
        'stop_loss_atr': float(args.stop_loss_atr),
        'stop_padding_pct': float(args.stop_padding_pct),
        'take_profit_rr': float(args.take_profit_rr),
        'use_dynamic_rr': bool(args.use_dynamic_rr),
        'use_calibration': bool(args.use_calibration),
        'cooldown_bars_after_stop': int(args.cooldown_bars_after_stop),
        'trade_side': str(args.trade_side),
        'touch_threshold_atr': float(args.touch_threshold_atr),
        'ema_touch_mode': str(args.ema_touch_mode),
        'base_timeframe_idx': int(cfg.base_timeframe_idx),
        'base_timeframe': base_tf,
        'ema_periods': list(cfg.features.ema_periods),
        'pullback_ema': int(cfg.labels.pullback_ema),
        'pullback_threshold_atr': float(cfg.labels.pullback_threshold),
        'test_bars': int(len(test)),
        'test_start_index': int(start),
    }
    paths = save_backtest_logs(
        res,
        cfg,
        Path('./backtests-logs-at-close'),
        model_dir=cfg.model.model_dir,
        driver=Path(__file__).name,
        parameters=log_params,
        extra_metrics=stop_stats,
    )
    print(f"Saved backtest logs -> {paths['summary']} and {paths['trades']}")


if __name__ == '__main__':
    main()

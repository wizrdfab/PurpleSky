"""
Training pipeline for TrendFollower ML system.
Orchestrates data loading, feature engineering, labeling, and model training.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import time

from config import TrendFollowerConfig, DEFAULT_CONFIG
from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from labels import create_training_dataset, detect_pullback_zones
from models import TrendFollowerModels
from diagnostic_logger import DiagnosticLogger


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (no shuffling).
    
    Args:
        df: DataFrame to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        Tuple of (train, val, test) DataFrames
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def run_training_pipeline(
    config: TrendFollowerConfig,
    enable_diagnostics: bool = True,
    two_pass: bool = False,
    use_noise_filtering: bool = False,
    use_seed_ensemble: bool = False,
    n_ensemble_seeds: int = 5,
) -> Dict:
    """
    Run the complete training pipeline.

    Args:
        config: Complete configuration
        enable_diagnostics: Whether to create diagnostic logs
        two_pass: Whether to use two-pass training (train with val, then retrain on train+val)
        use_noise_filtering: If True, filter out features ranking below random noise
        use_seed_ensemble: If True, train N models with different seeds and average
        n_ensemble_seeds: Number of seeds to use for ensembling (default: 5)

    Returns:
        Dictionary with training results and metrics
    """
    results = {}
    
    # Initialize diagnostic logger
    diag = DiagnosticLogger('./logs') if enable_diagnostics else None
    
    print("=" * 60)
    print("TREND FOLLOWER TRAINING PIPELINE")
    print("=" * 60)
    
    # --- Step 1: Load Data ---
    print("\n[1/6] Loading trade data...")
    if diag:
        diag.log_section("1. Data Loading")
    start_time = time.time()
    
    # Get sample rate from config (default to 1.0 if not set)
    sample_rate = getattr(config, 'sample_rate', 1.0)
    
    trades = load_trades(config.data, sample_rate=sample_rate)
    trades = preprocess_trades(trades, config.data)
    
    results['total_trades'] = len(trades)
    print(f"      Loaded {len(trades):,} trades in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_metric("total_trades", len(trades))
        diag.log_metric("sample_rate", sample_rate)
        diag.log_dataframe_stats("trades", trades)
    
    # --- Step 2: Create Bars ---
    print("\n[2/6] Creating multi-timeframe bars...")
    if diag:
        diag.log_section("2. Bar Creation")
    start_time = time.time()
    
    bars_dict = create_multi_timeframe_bars(
        trades,
        config.features.timeframes,
        config.features.timeframe_names,
        config.data
    )
    
    base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    results['base_timeframe'] = base_tf
    results['bars_per_tf'] = {tf: len(bars) for tf, bars in bars_dict.items()}
    
    print(f"      Created bars in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_metric("base_timeframe", base_tf)
        for tf, bars in bars_dict.items():
            diag.log_metric(f"bars_{tf}", len(bars))
            if tf == base_tf:
                diag.log_dataframe_stats(f"bars_{tf}", bars)
    
    # --- Step 3: Calculate Features ---
    print("\n[3/6] Calculating features...")
    if diag:
        diag.log_section("3. Feature Calculation")
    start_time = time.time()
    
    featured_data = calculate_multi_timeframe_features(
        bars_dict,
        base_tf,
        config.features
    )
    
    feature_cols = get_feature_columns(featured_data)
    results['num_features'] = len(feature_cols)
    
    print(f"      Calculated {len(feature_cols)} features in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_metric("num_features", len(feature_cols))
        diag.log_feature_columns(feature_cols, featured_data)
        diag.log_dataframe_stats("featured_data", featured_data, show_columns=False)
    
    # --- Step 4: Generate Labels ---
    print("\n[4/6] Generating labels...")
    if diag:
        diag.log_section("4. Label Generation")
    start_time = time.time()
    
    labeled_data, feature_cols = create_training_dataset(
        featured_data,
        config.labels,
        config.features,
        base_tf
    )
    
    results['labeled_samples'] = len(labeled_data)
    print(f"      Generated labels in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_metric("labeled_samples", len(labeled_data))
        diag.log_metric("final_feature_count", len(feature_cols))
        
        # Log label distributions
        if 'trend_label' in labeled_data.columns:
            diag.log_label_distribution("trend_label", labeled_data['trend_label'])
        if 'regime' in labeled_data.columns:
            diag.log_label_distribution("regime", labeled_data['regime'])
        if 'pullback_success' in labeled_data.columns:
            pullback_labels = labeled_data['pullback_success'].dropna()
            if len(pullback_labels) > 0:
                diag.log_label_distribution("pullback_success", pullback_labels)
        
        # Re-check feature columns for leakage after labeling
        diag.log_feature_columns(feature_cols, labeled_data)
    
    # --- Step 5: Split Data ---
    print("\n[5/6] Splitting data...")
    if diag:
        diag.log_section("5. Data Split")
    
    train_df, val_df, test_df = time_series_split(
        labeled_data,
        config.model.train_ratio,
        config.model.val_ratio,
        config.model.test_ratio
    )
    
    results['train_size'] = len(train_df)
    results['val_size'] = len(val_df)
    results['test_size'] = len(test_df)
    
    print(f"      Train: {len(train_df):,} samples")
    print(f"      Val:   {len(val_df):,} samples")
    print(f"      Test:  {len(test_df):,} samples")
    
    if diag:
        diag.log_train_val_test_split(train_df, val_df, test_df, 'bar_time')
    
    # Prepare feature matrices
    # Fill NaN with 0 for features (after forward fill)
    X_train = train_df[feature_cols].fillna(0)
    X_val = val_df[feature_cols].fillna(0) if len(val_df) else None
    X_test = test_df[feature_cols].fillna(0) if len(test_df) else None
    
    if diag:
        # Check feature-label correlations for leakage detection
        diag.log_section("5b. Leakage Detection - Feature-Label Correlations")
        diag.log_feature_label_correlations(X_train, train_df['trend_label'], 'trend_label')
        if 'regime' in train_df.columns:
            diag.log_feature_label_correlations(X_train, train_df['regime'], 'regime')
    
    # --- Step 6: Train Models ---
    print("\n[6/6] Training models...")
    if diag:
        diag.log_section("6. Model Training")
    
    models = TrendFollowerModels(config.model)
    
    # Train Trend Classifier
    print("\n  Training Trend Classifier...")
    start_time = time.time()
    
    y_trend_train = train_df['trend_label']
    y_trend_val = val_df['trend_label'] if len(val_df) else None
    
    trend_metrics = models.trend_classifier.train(
        X_train, y_trend_train,
        X_val, y_trend_val,
        verbose=True
    )
    
    results['trend_classifier'] = trend_metrics
    print(f"    Train Accuracy: {trend_metrics['train_accuracy']:.3f}")
    if 'val_accuracy' in trend_metrics:
        print(f"    Val Accuracy:   {trend_metrics['val_accuracy']:.3f}")
    print(f"    Trained in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_model_training("TrendClassifier", trend_metrics)
        # Log feature importance
        if hasattr(models.trend_classifier.model, 'feature_importances_'):
            importances = dict(zip(feature_cols, models.trend_classifier.model.feature_importances_))
            diag.log_feature_importance("TrendClassifier", importances)
    
    # Train Entry Quality Model (only on pullback zones)
    # Now with: sequential TP/SL labeling, multi-tier quality, probability calibration
    print("\n  Training Entry Quality Model...")
    start_time = time.time()

    # Get pullback samples
    pullback_mask_train = ~train_df['pullback_success'].isna()
    pullback_mask_val = (~val_df['pullback_success'].isna()) if len(val_df) else None

    if diag:
        diag.log_metric("pullback_samples_train", int(pullback_mask_train.sum()))
        diag.log_metric("pullback_samples_val", int(pullback_mask_val.sum()) if pullback_mask_val is not None else 0)

        # Log new sequential labeling stats
        if 'pullback_exit_type' in train_df.columns:
            exit_types = train_df.loc[pullback_mask_train, 'pullback_exit_type'].value_counts()
            diag.log_raw(f"\n  Pullback Exit Types (sequential TP/SL):\n")
            for exit_type, count in exit_types.items():
                diag.log_raw(f"    {exit_type}: {count}\n")

        if 'pullback_tier' in train_df.columns:
            tier_dist = train_df.loc[pullback_mask_train, 'pullback_tier'].dropna().value_counts().sort_index()
            diag.log_raw(f"\n  Pullback Quality Tiers:\n")
            tier_names = {0: 'Stopped', 1: 'BE-1R', 2: '1R-2R', 3: '2R+'}
            for tier, count in tier_dist.items():
                diag.log_raw(f"    Tier {int(tier)} ({tier_names.get(int(tier), '?')}): {count}\n")

    if pullback_mask_train.sum() > 100:
        X_entry_train = X_train[pullback_mask_train]
        y_success_train = train_df.loc[pullback_mask_train, 'pullback_success'].astype(int)
        y_rr_train = train_df.loc[pullback_mask_train, 'pullback_rr']

        # Get tier labels for multi-tier quality classifier
        y_tier_train = None
        if 'pullback_tier' in train_df.columns:
            y_tier_train = train_df.loc[pullback_mask_train, 'pullback_tier']

        has_val_pullbacks = pullback_mask_val is not None and int(pullback_mask_val.sum()) > 0 and X_val is not None
        X_entry_val = X_val[pullback_mask_val] if has_val_pullbacks else None
        y_success_val = val_df.loc[pullback_mask_val, 'pullback_success'].astype(int) if has_val_pullbacks else None
        y_rr_val = val_df.loc[pullback_mask_val, 'pullback_rr'] if has_val_pullbacks else None
        y_tier_val = val_df.loc[pullback_mask_val, 'pullback_tier'] if (has_val_pullbacks and 'pullback_tier' in val_df.columns) else None

        if diag:
            diag.log_label_distribution("pullback_success_train", y_success_train)
            diag.log_feature_label_correlations(X_entry_train, y_success_train, 'pullback_success')

        entry_metrics = models.entry_model.train(
            X_entry_train, y_success_train, y_rr_train,
            X_entry_val, y_success_val, y_rr_val,
            y_tier_train=y_tier_train,
            y_tier_val=y_tier_val,
            verbose=True,
            calibrate=True,  # Enable probability calibration
            use_noise_filtering=use_noise_filtering,
            use_seed_ensemble=use_seed_ensemble,
            n_ensemble_seeds=n_ensemble_seeds,
        )

        results['entry_model'] = entry_metrics
        print(f"    Train Accuracy:  {entry_metrics['train_accuracy']:.3f}")
        print(f"    Train Precision: {entry_metrics['train_precision']:.3f}")
        if 'val_accuracy' in entry_metrics:
            print(f"    Val Accuracy:    {entry_metrics['val_accuracy']:.3f}")
            print(f"    Val Precision:   {entry_metrics['val_precision']:.3f}")

        # Log calibration results
        if 'calibration' in entry_metrics:
            cal = entry_metrics['calibration']
            print(f"    Pre-calibration ECE:  {cal['pre_calibration_ece']:.4f}")
            print(f"    Post-calibration ECE: {cal['post_calibration_ece']:.4f}")
            print(f"    ECE Improvement:      {cal['ece_improvement']:.4f}")

        if diag:
            diag.log_model_training("EntryQualityModel", entry_metrics)
            if hasattr(models.entry_model, 'classifier') and hasattr(models.entry_model.classifier, 'feature_importances_'):
                importances = dict(zip(feature_cols, models.entry_model.classifier.feature_importances_))
                diag.log_feature_importance("EntryQualityModel", importances)

            # Log calibration details
            if 'calibration' in entry_metrics:
                diag.log_section("6b. Probability Calibration")
                cal = entry_metrics['calibration']
                diag.log_metric("pre_calibration_ece", cal['pre_calibration_ece'])
                diag.log_metric("post_calibration_ece", cal['post_calibration_ece'])
                diag.log_metric("ece_improvement", cal['ece_improvement'])

                diag.log_raw("\n  Calibration Bin Details (Post-Calibration):\n")
                diag.log_raw(f"    {'Bin':<12} {'Count':<8} {'Confidence':<12} {'Actual':<12} {'Error':<10}\n")
                diag.log_raw(f"    {'-'*54}\n")
                for bin_info in cal['post_calibration_details']['bin_details']:
                    diag.log_raw(
                        f"    {bin_info['bin']:<12} "
                        f"{bin_info['count']:<8} "
                        f"{bin_info['avg_confidence']:<12.3f} "
                        f"{bin_info['actual_accuracy']:<12.3f} "
                        f"{bin_info['calibration_error']:<10.3f}\n"
                    )
    else:
        print(f"    Warning: Only {pullback_mask_train.sum()} pullback samples, skipping entry model")
        results['entry_model'] = {'skipped': True}
        if diag:
            diag.log_warning(f"Entry model skipped - only {pullback_mask_train.sum()} pullback samples")

    print(f"    Trained in {time.time() - start_time:.1f}s")
    
    # Train Regime Classifier
    print("\n  Training Regime Classifier...")
    if diag:
        diag.log_section("6c. Regime Classifier Training")
    start_time = time.time()
    
    y_regime_train = train_df['regime']
    y_regime_val = val_df['regime'] if len(val_df) else None
    
    if diag:
        diag.log_label_distribution("regime_train", y_regime_train)
        diag.log_feature_label_correlations(X_train, y_regime_train, 'regime')
        
        # Special check: explain why regime classifier gets 100%
        adx_col = f'{base_tf}_adx'
        align_col = f'{base_tf}_ema_alignment'
        if adx_col in X_train.columns and align_col in X_train.columns:
            diag.log_raw(f"\n  NOTE: Regime is a deterministic function of {adx_col} and {align_col}\n")
            diag.log_raw(f"  This is expected to achieve ~100% accuracy (not leakage, just trivial task)\n")
    
    regime_metrics = models.regime_classifier.train(
        X_train, y_regime_train,
        X_val, y_regime_val,
        verbose=True
    )
    
    results['regime_classifier'] = regime_metrics
    print(f"    Train Accuracy: {regime_metrics['train_accuracy']:.3f}")
    if 'val_accuracy' in regime_metrics:
        print(f"    Val Accuracy:   {regime_metrics['val_accuracy']:.3f}")
    print(f"    Trained in {time.time() - start_time:.1f}s")
    
    if diag:
        diag.log_model_training("RegimeClassifier", regime_metrics)
        if hasattr(models.regime_classifier.model, 'feature_importances_'):
            importances = dict(zip(feature_cols, models.regime_classifier.model.feature_importances_))
            diag.log_feature_importance("RegimeClassifier", importances)
    
    # --- Optional: Two-pass retrain on Train+Val ---
    # Pass 1 trains with a validation split (early stopping). Pass 2 retrains on Train+Val
    # using the best iteration counts found in pass 1 (no validation). The final models
    # are then saved and evaluated on the test split (if any).
    if two_pass:
        if len(val_df) == 0:
            raise ValueError("two_pass=True requires a non-empty validation split (val_ratio > 0).")

        results['pass1_trend_classifier'] = results.get('trend_classifier')
        results['pass1_entry_model'] = results.get('entry_model')
        results['pass1_regime_classifier'] = results.get('regime_classifier')

        entry_pass1 = results.get('entry_model') if isinstance(results.get('entry_model'), dict) else {}
        best_iters = {
            'trend_classifier': (results.get('trend_classifier') or {}).get('best_iteration'),
            'entry_model': entry_pass1.get('best_iteration'),
            'regime_classifier': (results.get('regime_classifier') or {}).get('best_iteration'),
        }
        results['two_pass'] = {
            'enabled': True,
            'best_iterations': best_iters,
            'train_plus_val_size': int(len(train_df) + len(val_df)),
        }

        print("\n  Two-pass retrain (final models on Train+Val):")
        print(f"      Train+Val size: {len(train_df) + len(val_df):,} samples")
        print(
            "      Best iterations (pass 1): "
            f"trend={best_iters['trend_classifier']}, "
            f"entry={best_iters['entry_model']}, "
            f"regime={best_iters['regime_classifier']}"
        )

        full_df = pd.concat([train_df, val_df])
        X_full = full_df[feature_cols].fillna(0)

        # Trend classifier (final)
        print("\n  Retraining Trend Classifier on Train+Val...")
        start_time = time.time()
        trend_metrics = models.trend_classifier.train(
            X_full,
            full_df['trend_label'],
            None,
            None,
            verbose=True,
            n_estimators=best_iters['trend_classifier'],
        )
        results['trend_classifier'] = trend_metrics
        print(f"    Train Accuracy: {trend_metrics['train_accuracy']:.3f}")
        print(f"    Trained in {time.time() - start_time:.1f}s")

        # Entry quality model (final)
        print("\n  Retraining Entry Quality Model on Train+Val...")
        start_time = time.time()
        pullback_mask_full = ~full_df['pullback_success'].isna()
        if pullback_mask_full.sum() > 100:
            X_entry_full = X_full[pullback_mask_full]
            y_success_full = full_df.loc[pullback_mask_full, 'pullback_success'].astype(int)
            y_rr_full = full_df.loc[pullback_mask_full, 'pullback_rr']

            entry_metrics = models.entry_model.train(
                X_entry_full,
                y_success_full,
                y_rr_full,
                None,
                None,
                None,
                verbose=True,
                n_estimators=best_iters['entry_model'],
                use_noise_filtering=use_noise_filtering,
                use_seed_ensemble=use_seed_ensemble,
                n_ensemble_seeds=n_ensemble_seeds,
            )
            results['entry_model'] = entry_metrics
            print(f"    Train Accuracy:  {entry_metrics['train_accuracy']:.3f}")
            print(f"    Train Precision: {entry_metrics['train_precision']:.3f}")
        else:
            print(f"    Warning: Only {pullback_mask_full.sum()} pullback samples, skipping entry model")
            results['entry_model'] = {'skipped': True}
        print(f"    Trained in {time.time() - start_time:.1f}s")

        # Regime classifier (final)
        print("\n  Retraining Regime Classifier on Train+Val...")
        start_time = time.time()
        regime_metrics = models.regime_classifier.train(
            X_full,
            full_df['regime'],
            None,
            None,
            verbose=True,
            n_estimators=best_iters['regime_classifier'],
        )
        results['regime_classifier'] = regime_metrics
        print(f"    Train Accuracy: {regime_metrics['train_accuracy']:.3f}")
        print(f"    Trained in {time.time() - start_time:.1f}s")

    # --- Save Models ---
    print("\n  Saving models...")
    models.save_all(config.model.model_dir)
    
    # --- Test Set Evaluation ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    if diag:
        diag.log_section("7. Test Set Evaluation")
    
    # Trend classifier
    test_trend_acc = None
    if X_test is None or len(test_df) == 0:
        print("\n  No test split (test_ratio=0). Skipping test-set evaluation.")
    else:
        trend_pred = models.trend_classifier.predict(X_test)
        test_trend_acc = (trend_pred['prediction'] == test_df['trend_label'].values).mean()
        print(f"\n  Trend Classifier Test Accuracy: {test_trend_acc:.3f}")
        
        if diag:
            diag.log_metric("trend_classifier_test_accuracy", test_trend_acc, warn_if='high')
            diag.log_sample_predictions(
                "TrendClassifier",
                X_test,
                test_df['trend_label'],
                trend_pred['prediction'],
                n_samples=20,
            )
        
        # Per-class accuracy
        for label, name in [(-1, 'Down'), (0, 'Neutral'), (1, 'Up')]:
            mask = test_df['trend_label'].values == label
            if mask.sum() > 0:
                acc = (trend_pred['prediction'][mask] == label).mean()
                print(f"    {name}: {acc:.3f} (n={mask.sum()})")
                if diag:
                    diag.log_metric(f"trend_class_{name}_accuracy", acc)
    
    results['test_trend_accuracy'] = test_trend_acc
    
    # Feature importance
    print("\n  Top 20 Most Important Features (Trend Classifier):")
    importance = models.trend_classifier.feature_importance.head(20)
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.0f}")
    
    results['feature_importance'] = importance.to_dict('records')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save diagnostic log
    if diag:
        diag.log_section("8. Final Summary")
        if test_trend_acc is not None:
            diag.log_metric("final_test_accuracy", test_trend_acc)
        
        # Add all results to log
        for k, v in results.items():
            if isinstance(v, (int, float, str)):
                diag.log_metric(f"result_{k}", v)
        
        log_path = diag.save()
        results['diagnostic_log'] = str(log_path)
    
    return results, models, test_df, X_test, feature_cols


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Train TrendFollower models')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing trade CSV files')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--base-tf', type=int, default=1,
                       help='Index of base timeframe (0=1m, 1=5m, 2=15m, 3=1h, 4=4h)')
    
    args = parser.parse_args()
    
    # Create config
    config = DEFAULT_CONFIG
    config.data.data_dir = Path(args.data_dir)
    config.model.model_dir = Path(args.model_dir)
    config.base_timeframe_idx = args.base_tf
    
    # Run training
    results = run_training_pipeline(config)
    
    # Save results
    results_file = Path(args.model_dir) / 'training_results.json'
    
    # Convert non-serializable items
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, list, dict)):
            serializable_results[k] = v
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            serializable_results[k] = v.to_dict('records')
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

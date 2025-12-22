"""
Diagnostic Logger for TrendFollower Training Pipeline

This module creates detailed logs that capture the inner workings of the training
process, allowing for later analysis and debugging.

The log includes:
- Data statistics at each stage
- Feature distributions and correlations with labels
- Model training details
- Potential leakage detection
- Sample predictions for sanity checks
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json


class DiagnosticLogger:
    """
    Comprehensive diagnostic logger for training pipeline.
    
    Usage:
        logger = DiagnosticLogger('./logs')
        logger.log_section("Data Loading")
        logger.log_metric("total_trades", 1000000)
        logger.log_dataframe_stats("trades", df)
        logger.save()
    """
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_diagnostic_{self.timestamp}.txt'
        self.json_file = self.log_dir / f'training_diagnostic_{self.timestamp}.json'
        
        self.sections: List[Dict] = []
        self.current_section: Optional[Dict] = None
        self.all_metrics: Dict[str, Any] = {}
        
        self._write_header()
    
    def _write_header(self):
        """Write log header"""
        header = f"""
{'='*80}
TRENDFOLOWER TRAINING DIAGNOSTIC LOG
{'='*80}
Timestamp: {datetime.now().isoformat()}
Log File:  {self.log_file}
{'='*80}

PURPOSE: This log captures internal state during training to help identify:
- Data leakage (features that shouldn't be used)
- Label distribution issues
- Feature-label correlations that are too high (suspicious)
- Model behavior anomalies
- Any other issues with the training pipeline

"""
        with open(self.log_file, 'w') as f:
            f.write(header)
    
    def log_section(self, name: str):
        """Start a new section in the log"""
        if self.current_section:
            self.sections.append(self.current_section)
        
        self.current_section = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'warnings': [],
            'data_samples': {}
        }
        
        section_header = f"\n\n{'='*80}\n[SECTION] {name}\n{'='*80}\nTime: {datetime.now().isoformat()}\n"
        self._append(section_header)
    
    def log_metric(self, name: str, value: Any, warn_if: Optional[str] = None):
        """Log a single metric"""
        if self.current_section:
            self.current_section['metrics'][name] = value
        self.all_metrics[name] = value
        
        # Check for warnings
        warning = None
        if warn_if:
            if warn_if == 'perfect' and value == 1.0:
                warning = f"⚠️ WARNING: {name} = {value} (perfect score - possible leakage!)"
            elif warn_if == 'zero' and value == 0.0:
                warning = f"⚠️ WARNING: {name} = {value} (zero - something wrong?)"
            elif warn_if == 'high' and isinstance(value, (int, float)) and value > 0.95:
                warning = f"⚠️ WARNING: {name} = {value} (suspiciously high)"
        
        line = f"  {name}: {value}"
        if warning:
            line += f"\n  {warning}"
            if self.current_section:
                self.current_section['warnings'].append(warning)
        
        self._append(line + "\n")
    
    def log_dataframe_stats(self, name: str, df: pd.DataFrame, show_columns: bool = True):
        """Log statistics about a DataFrame"""
        stats = {
            'shape': df.shape,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'null_counts': df.isnull().sum().sum(),
            'null_columns': df.columns[df.isnull().any()].tolist()[:10],  # First 10
        }
        
        text = f"\n  DataFrame: {name}\n"
        text += f"    Shape: {stats['shape'][0]:,} rows x {stats['shape'][1]} columns\n"
        text += f"    Memory: {stats['memory_mb']:.2f} MB\n"
        text += f"    Null values: {stats['null_counts']:,}\n"
        
        if stats['null_columns']:
            text += f"    Columns with nulls: {stats['null_columns']}\n"
        
        if show_columns and len(df.columns) <= 50:
            text += f"    Columns: {list(df.columns)}\n"
        elif show_columns:
            text += f"    First 20 columns: {list(df.columns[:20])}\n"
            text += f"    ... and {len(df.columns) - 20} more\n"
        
        if self.current_section:
            self.current_section['metrics'][f'{name}_stats'] = stats
        
        self._append(text)
    
    def log_feature_columns(self, feature_cols: List[str], df: pd.DataFrame):
        """Log feature columns and check for suspicious ones"""
        self._append(f"\n  Feature Columns ({len(feature_cols)} total):\n")
        
        # Check for suspicious column names that might indicate leakage
        suspicious_patterns = [
            'label', 'target', 'outcome', 'success', 'trend_label', 'regime',
            'pullback_success', 'mfe', 'mae', 'max_favorable', 'max_adverse',
            'future', 'forward', '_y', '_target'
        ]
        
        suspicious_found = []
        for col in feature_cols:
            for pattern in suspicious_patterns:
                if pattern.lower() in col.lower():
                    suspicious_found.append((col, pattern))
        
        if suspicious_found:
            self._append(f"\n  ⚠️ POTENTIAL LEAKAGE - Suspicious feature columns found:\n")
            for col, pattern in suspicious_found:
                self._append(f"    - {col} (contains '{pattern}')\n")
            if self.current_section:
                self.current_section['warnings'].append(f"Suspicious features: {suspicious_found}")
        else:
            self._append(f"  ✓ No suspicious feature names detected\n")
        
        # Log first 30 feature columns
        self._append(f"\n  First 30 features: {feature_cols[:30]}\n")
        if len(feature_cols) > 30:
            self._append(f"  ... and {len(feature_cols) - 30} more\n")
    
    def log_label_distribution(self, name: str, labels: pd.Series):
        """Log distribution of labels"""
        dist = labels.value_counts().sort_index()
        dist_pct = labels.value_counts(normalize=True).sort_index()
        
        text = f"\n  Label Distribution: {name}\n"
        text += f"    Total samples: {len(labels):,}\n"
        for val in dist.index:
            text += f"    Class {val}: {dist[val]:,} ({dist_pct[val]:.1%})\n"
        
        # Check for severe imbalance
        if dist_pct.max() > 0.8:
            warning = f"⚠️ WARNING: Severe class imbalance in {name} - majority class is {dist_pct.max():.1%}"
            text += f"    {warning}\n"
            if self.current_section:
                self.current_section['warnings'].append(warning)
        
        if self.current_section:
            self.current_section['metrics'][f'{name}_distribution'] = dist.to_dict()
        
        self._append(text)
    
    def log_feature_label_correlations(self, X: pd.DataFrame, y: pd.Series, 
                                        label_name: str, top_n: int = 20):
        """Log correlations between features and labels - high correlation might indicate leakage"""
        self._append(f"\n  Feature-Label Correlations ({label_name}):\n")
        
        correlations = {}
        for col in X.columns:
            try:
                corr = X[col].corr(y.astype(float))
                if pd.notna(corr):
                    correlations[col] = abs(corr)
            except:
                pass
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Check for suspiciously high correlations
        suspicious = [(col, corr) for col, corr in sorted_corr if corr > 0.9]
        if suspicious:
            self._append(f"\n  ⚠️ POTENTIAL LEAKAGE - Features with correlation > 0.9:\n")
            for col, corr in suspicious:
                self._append(f"    - {col}: {corr:.4f}\n")
            if self.current_section:
                self.current_section['warnings'].append(f"High correlation features for {label_name}: {suspicious}")
        
        # Log top N correlations
        self._append(f"\n  Top {top_n} correlations:\n")
        for col, corr in sorted_corr[:top_n]:
            flag = " ⚠️" if corr > 0.8 else ""
            self._append(f"    {col}: {corr:.4f}{flag}\n")
        
        if self.current_section:
            self.current_section['metrics'][f'{label_name}_top_correlations'] = dict(sorted_corr[:top_n])
    
    def log_train_val_test_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, time_col: str = 'bar_time'):
        """Log details about data split"""
        self._append(f"\n  Train/Val/Test Split:\n")
        
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            self._append(f"    {name}: {len(df):,} samples\n")
            if time_col in df.columns:
                min_time = pd.to_datetime(df[time_col].min(), unit='s')
                max_time = pd.to_datetime(df[time_col].max(), unit='s')
                self._append(f"      Time range: {min_time} to {max_time}\n")
        
        # Check for time overlap (would indicate leakage)
        if time_col in train_df.columns and time_col in test_df.columns:
            train_max = train_df[time_col].max()
            test_min = test_df[time_col].min()
            
            if train_max >= test_min:
                warning = f"⚠️ CRITICAL: Train/Test time overlap detected! train_max={train_max}, test_min={test_min}"
                self._append(f"    {warning}\n")
                if self.current_section:
                    self.current_section['warnings'].append(warning)
            else:
                self._append(f"    ✓ No time overlap between train and test\n")
    
    def log_model_training(self, model_name: str, metrics: Dict[str, Any]):
        """Log model training results"""
        self._append(f"\n  Model: {model_name}\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self._append(f"    {key}: {value:.4f}\n")
                
                # Check for perfect scores
                if 'accuracy' in key.lower() or 'precision' in key.lower():
                    if value >= 0.99:
                        warning = f"⚠️ WARNING: {model_name} {key} = {value:.4f} (suspiciously high)"
                        self._append(f"    {warning}\n")
                        if self.current_section:
                            self.current_section['warnings'].append(warning)
            else:
                self._append(f"    {key}: {value}\n")
        
        if self.current_section:
            self.current_section['metrics'][f'{model_name}_metrics'] = metrics
    
    def log_feature_importance(self, model_name: str, importances: Dict[str, float], top_n: int = 20):
        """Log feature importance from a model"""
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        self._append(f"\n  Feature Importance ({model_name}) - Top {top_n}:\n")
        for i, (col, imp) in enumerate(sorted_imp[:top_n]):
            self._append(f"    {i+1}. {col}: {imp:.0f}\n")
        
        if self.current_section:
            self.current_section['metrics'][f'{model_name}_top_features'] = dict(sorted_imp[:top_n])
    
    def log_sample_predictions(self, model_name: str, X: pd.DataFrame, y_true: pd.Series, 
                               predictions: np.ndarray, n_samples: int = 10):
        """Log sample predictions for sanity check"""
        self._append(f"\n  Sample Predictions ({model_name}) - First {n_samples}:\n")
        self._append(f"    {'Index':<10} {'True':<10} {'Predicted':<10} {'Correct':<10}\n")
        self._append(f"    {'-'*40}\n")
        
        for i in range(min(n_samples, len(y_true))):
            true_val = y_true.iloc[i]
            pred_val = predictions[i]
            correct = "✓" if true_val == pred_val else "✗"
            self._append(f"    {i:<10} {true_val:<10} {pred_val:<10} {correct:<10}\n")
    
    def log_backtest_trades(self, trades: List[Dict]):
        """Log details of backtest trades"""
        self._append(f"\n  Backtest Trades ({len(trades)} total):\n")
        
        if not trades:
            self._append("    No trades executed\n")
            return
        
        self._append(f"    {'#':<4} {'Dir':<6} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'Result':<8} {'Grade':<6}\n")
        self._append(f"    {'-'*70}\n")
        
        for i, trade in enumerate(trades[:20]):  # First 20 trades
            direction = "LONG" if trade.get('direction', 0) > 0 else "SHORT"
            entry = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0)
            result = "WIN" if pnl > 0 else "LOSS"
            grade = trade.get('quality', '?')
            
            self._append(f"    {i+1:<4} {direction:<6} {entry:<12.6f} {exit_price:<12.6f} ${pnl:<11.2f} {result:<8} {grade:<6}\n")
        
        if len(trades) > 20:
            self._append(f"    ... and {len(trades) - 20} more trades\n")
    
    def log_warning(self, message: str):
        """Log a warning"""
        warning = f"⚠️ WARNING: {message}"
        self._append(f"\n  {warning}\n")
        if self.current_section:
            self.current_section['warnings'].append(warning)
    
    def log_error(self, message: str):
        """Log an error"""
        error = f"❌ ERROR: {message}"
        self._append(f"\n  {error}\n")
        if self.current_section:
            self.current_section['warnings'].append(error)
    
    def log_raw(self, text: str):
        """Log raw text"""
        self._append(text)
    
    def _append(self, text: str):
        """Append text to log file"""
        with open(self.log_file, 'a', encoding='utf8') as f:
            f.write(text)
    
    def save(self):
        """Save the log and create JSON summary"""
        if self.current_section:
            self.sections.append(self.current_section)
        
        # Write summary
        summary = f"""

{'='*80}
SUMMARY
{'='*80}

Total Sections: {len(self.sections)}
"""
        # Collect all warnings
        all_warnings = []
        for section in self.sections:
            all_warnings.extend(section.get('warnings', []))
        
        if all_warnings:
            summary += f"\n⚠️ WARNINGS FOUND ({len(all_warnings)}):\n"
            for w in all_warnings:
                summary += f"  - {w}\n"
        else:
            summary += "\n✓ No warnings detected\n"
        
        summary += f"\nLog saved to: {self.log_file}\n"
        summary += f"JSON saved to: {self.json_file}\n"
        
        self._append(summary)
        
        # Save JSON with proper serialization
        def make_serializable(obj):
            """Convert non-serializable objects to serializable format"""
            if isinstance(obj, dict):
                # Convert dict keys to strings and recursively process values
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'dtype'):  # Handle pandas dtypes
                return str(obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        json_data = {
            'timestamp': self.timestamp,
            'sections': make_serializable(self.sections),
            'all_metrics': make_serializable(self.all_metrics),
            'all_warnings': all_warnings
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nDiagnostic log saved to: {self.log_file}")
        return self.log_file


# Global logger instance
_logger: Optional[DiagnosticLogger] = None


def get_logger(log_dir: str = './logs') -> DiagnosticLogger:
    """Get or create the global diagnostic logger"""
    global _logger
    if _logger is None:
        _logger = DiagnosticLogger(log_dir)
    return _logger


def reset_logger():
    """Reset the global logger"""
    global _logger
    _logger = None

use std::path::Path;

use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyBytes, PyDict, PyModule};

mod pipeline;

#[pyfunction]
fn run_pipeline(config_path: &str, output_dir: &str, write_intermediate: Option<bool>) -> PyResult<Py<PyDict>> {
    let config_path = Path::new(config_path);
    let output_dir = Path::new(output_dir);
    let mut config = pipeline::load_config(config_path)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::resolve_data_dir(&mut config, config_path)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::run_pipeline(&config, output_dir, write_intermediate.unwrap_or(false))
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

    let feature_cols_path = output_dir.join("feature_cols.json");
    let raw = std::fs::read_to_string(&feature_cols_path)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let feature_cols: Vec<String> = serde_json::from_str(&raw)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let dataset_path = output_dir.join("dataset.arrow");

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("dataset_path", dataset_path.to_string_lossy().to_string())?;
        dict.set_item("feature_cols_path", feature_cols_path.to_string_lossy().to_string())?;
        dict.set_item("feature_cols", feature_cols)?;
        Ok(dict.unbind())
    })
}

#[pyfunction]
fn run_pipeline_memory(config_path: &str) -> PyResult<Py<PyDict>> {
    let config_path = Path::new(config_path);
    let mut config = pipeline::load_config(config_path)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::resolve_data_dir(&mut config, config_path)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let (dataset_bytes, feature_cols) = pipeline::run_pipeline_in_memory(&config)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("dataset_bytes", PyBytes::new(py, &dataset_bytes))?;
        dict.set_item("feature_cols", feature_cols)?;
        Ok(dict.unbind())
    })
}

#[pyfunction]
fn run_pipeline_memory_json(config_json: &str) -> PyResult<Py<PyDict>> {
    let mut config = pipeline::load_config_from_str(config_json)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::resolve_data_dir(&mut config, Path::new("."))
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let (dataset_bytes, feature_cols) = pipeline::run_pipeline_in_memory(&config)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("dataset_bytes", PyBytes::new(py, &dataset_bytes))?;
        dict.set_item("feature_cols", feature_cols)?;
        Ok(dict.unbind())
    })
}

#[pyfunction]
fn run_bars_memory_json(config_json: &str, max_bars: Option<usize>) -> PyResult<Py<PyDict>> {
    let mut config = pipeline::load_config_from_str(config_json)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::resolve_data_dir(&mut config, Path::new("."))
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let bars = pipeline::build_bars_from_config(&config, max_bars)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        let bars_dict = PyDict::new(py);
        let counts_dict = PyDict::new(py);
        for (tf_name, frame) in bars {
            let bytes = pipeline::write_arrow_bytes(&frame)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
            bars_dict.set_item(tf_name.clone(), PyBytes::new(py, &bytes))?;
            counts_dict.set_item(tf_name, frame.len() as i64)?;
        }
        dict.set_item("bars_bytes", bars_dict)?;
        dict.set_item("bar_counts", counts_dict)?;
        Ok(dict.unbind())
    })
}

#[pyfunction]
fn init_trades_cache(config_json: &str) -> PyResult<Py<PyDict>> {
    let mut config = pipeline::load_config_from_str(config_json)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    pipeline::resolve_data_dir(&mut config, Path::new("."))
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let trades = pipeline::load_trades_cached(&config.data)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    let trade_count = trades.len();

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("trade_count", trade_count)?;
        Ok(dict.unbind())
    })
}

#[pyfunction]
fn clear_trades_cache() -> PyResult<()> {
    pipeline::clear_trades_cache();
    Ok(())
}

#[pyclass]
struct TradeIndex {
    config: pipeline::DataConfig,
    timeframe_seconds: i64,
}

#[pymethods]
impl TradeIndex {
    #[new]
    fn new(config_json: &str, timeframe_seconds: i64) -> PyResult<Self> {
        let mut config = pipeline::load_config_from_str(config_json)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        pipeline::resolve_data_dir(&mut config, Path::new("."))
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        pipeline::load_trades_index_cached(&config.data, timeframe_seconds)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        Ok(TradeIndex {
            config: config.data,
            timeframe_seconds,
        })
    }

    fn check_exit(&self, bar_time: i64, direction: i64, stop: f64, target: f64) -> PyResult<(i32, f64)> {
        pipeline::check_exit_in_bar(
            &self.config,
            self.timeframe_seconds,
            bar_time,
            direction,
            stop,
            target,
        )
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymodule]
fn sofia_rust_pipeline(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline_memory, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline_memory_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_bars_memory_json, m)?)?;
    m.add_function(wrap_pyfunction!(init_trades_cache, m)?)?;
    m.add_function(wrap_pyfunction!(clear_trades_cache, m)?)?;
    m.add_class::<TradeIndex>()?;
    Ok(())
}

"""
MLflow logging utilities for Slime distributed training.

Uses mlflow.tracking.MlflowClient for explicit control over experiments and runs.
This approach is preferred for distributed training because:
1. MlflowClient provides explicit experiment/run management without global state
2. Better suited for multi-process scenarios where each process needs to log to the same run
3. More control over tracking URI configuration per client instance
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Global state for the current run
_mlflow_client: Optional["MlflowClient"] = None
_current_run_id: Optional[str] = None
_tracking_uri: Optional[str] = None


def _get_mlflow_client() -> "MlflowClient":
    """
    Get or create the MlflowClient singleton.

    Returns:
        MlflowClient instance configured with the current tracking URI
    """
    global _mlflow_client, _tracking_uri

    if _mlflow_client is None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            raise ImportError(
                "mlflow is required for MLflow logging. "
                "Install with: pip install mlflow"
            )

        _mlflow_client = MlflowClient(tracking_uri=_tracking_uri)
        logger.info(f"Created MlflowClient with tracking_uri={_tracking_uri}")

    return _mlflow_client


def _validate_databricks_credentials() -> bool:
    """
    Validate that Databricks credentials are available.

    Checks for DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.

    Returns:
        True if credentials are available, False otherwise
    """
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host:
        logger.warning(
            "DATABRICKS_HOST environment variable not set. "
            "MLflow Databricks logging may fail."
        )
        return False

    if not token:
        logger.warning(
            "DATABRICKS_TOKEN environment variable not set. "
            "MLflow Databricks logging may fail."
        )
        return False

    # Mask token for logging (show first 4 chars only)
    masked_token = token[:4] + "..." if len(token) > 4 else "***"
    logger.info(f"Databricks credentials found: host={host}, token={masked_token}")
    return True


def _set_tracking_uri(tracking_uri: Optional[str]) -> str:
    """
    Set the MLflow tracking URI.

    Handles special case for 'databricks' URI and validates credentials.

    Args:
        tracking_uri: The tracking URI or 'databricks'

    Returns:
        The resolved tracking URI
    """
    global _tracking_uri, _mlflow_client

    if tracking_uri:
        if tracking_uri.lower() == "databricks":
            _tracking_uri = "databricks"
            # Validate Databricks credentials are available
            _validate_databricks_credentials()
        else:
            _tracking_uri = tracking_uri
    else:
        _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        # If MLFLOW_TRACKING_URI is databricks, validate credentials
        if _tracking_uri and _tracking_uri.lower() == "databricks":
            _validate_databricks_credentials()

    # Reset client to pick up new URI
    _mlflow_client = None

    return _tracking_uri


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict for MLflow params.

    MLflow params don't support nested structures, so we flatten with dot notation.

    Example:
        {'optimizer': {'lr': 1e-6}} -> {'optimizer.lr': 1e-6}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sanitize_metric_name(name: str) -> str:
    """
    Sanitize metric name for MLflow.

    MLflow allows: alphanumerics, underscores, dashes, periods, spaces, slashes.
    Replace any invalid characters.
    """
    # Most common names like "train/loss" are already valid
    return name.replace(":", "_")


def _compute_config_for_logging(args) -> Dict[str, Any]:
    """
    Prepare config dict for MLflow params logging.

    Filters to only include serializable values and adds environment context.
    """
    output = {}

    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            output[k] = v

    # Add relevant environment variables for reproducibility
    whitelist_env_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]
    for var in whitelist_env_vars:
        if var in os.environ:
            output[f"env.{var}"] = os.environ[var]

    return output


def _get_or_create_experiment(client: "MlflowClient", experiment_name: str) -> str:
    """
    Get existing experiment or create a new one.

    Args:
        client: MlflowClient instance
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        logger.info(f"Using existing experiment: {experiment_name} (id={experiment.experiment_id})")
        return experiment.experiment_id

    # Create new experiment
    experiment_id = client.create_experiment(experiment_name)
    logger.info(f"Created new experiment: {experiment_name} (id={experiment_id})")
    return experiment_id


def init_mlflow_primary(args, **kwargs) -> None:
    """
    Initialize MLflow on the primary rank (rank 0 in main train.py).

    Uses MlflowClient to:
    - Set up tracking URI
    - Get or create the experiment
    - Create a new run
    - Log initial params
    - Store run_id in args for secondary processes

    Args:
        args: Parsed arguments with MLflow configuration
        **kwargs: Additional keyword arguments (unused)
    """
    global _current_run_id

    if not getattr(args, "use_mlflow", False):
        args.mlflow_run_id = None
        return

    # Set tracking URI
    tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    resolved_uri = _set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {resolved_uri}")

    # Get client
    client = _get_mlflow_client()

    # Get or create experiment
    experiment_name = getattr(args, "mlflow_experiment_name", None)
    if not experiment_name:
        raise ValueError("--mlflow-experiment-name is required when using --use-mlflow")

    experiment_id = _get_or_create_experiment(client, experiment_name)

    # Generate run name
    run_name = getattr(args, "mlflow_run_name", None)
    if run_name:
        run_name = f"{run_name}_{uuid.uuid4().hex[:6]}"
    else:
        run_name = f"slime_run_{uuid.uuid4().hex[:6]}"

    # Create run using MlflowClient
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name
    )
    run_id = run.info.run_id
    _current_run_id = run_id

    logger.info(f"MLflow run created: {run_id} (name={run_name})")

    # Log params using client
    config = _compute_config_for_logging(args)
    flat_config = _flatten_dict(config)

    # MLflow has a limit on param value length (500 chars)
    # Log in batches to avoid issues with large param sets
    param_items = [(k, str(v)[:500]) for k, v in flat_config.items()]
    batch_size = 100

    for i in range(0, len(param_items), batch_size):
        batch = param_items[i:i + batch_size]
        for key, value in batch:
            try:
                client.log_param(run_id, key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    # Store run_id for secondary processes
    args.mlflow_run_id = run_id
    logger.info(f"MLflow primary initialized: run_id={run_id}")


def init_mlflow_secondary(args, **kwargs) -> None:
    """
    Initialize MLflow on secondary ranks (joins existing run).

    Secondary processes use the run_id from the primary to log to the same run.
    Uses MlflowClient for explicit run reference.

    Args:
        args: Parsed arguments with mlflow_run_id from primary
        **kwargs: Additional keyword arguments (unused)
    """
    global _current_run_id

    mlflow_run_id = getattr(args, "mlflow_run_id", None)
    if mlflow_run_id is None:
        return

    # Set tracking URI (same as primary)
    tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    _set_tracking_uri(tracking_uri)

    # Store run_id for logging
    _current_run_id = mlflow_run_id

    # Verify run exists
    client = _get_mlflow_client()
    try:
        run = client.get_run(mlflow_run_id)
        logger.info(f"MLflow secondary joined run: {mlflow_run_id} (status={run.info.status})")
    except Exception as e:
        logger.warning(f"Could not verify run {mlflow_run_id}: {e}")


def mlflow_log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """
    Log metrics to MLflow with step using MlflowClient.

    Handles:
    - Metric name sanitization
    - Type conversion (torch tensors to Python scalars)
    - Filtering non-numeric values
    - Batch logging for efficiency

    Args:
        metrics: Dictionary of metric names to values
        step: Training step number
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    timestamp = int(time.time() * 1000)  # MLflow expects milliseconds

    # Prepare metrics for batch logging
    try:
        from mlflow.entities import Metric
    except ImportError:
        logger.warning("mlflow.entities.Metric not available, skipping batch logging")
        return

    metric_list: List["Metric"] = []

    for k, v in metrics.items():
        # Convert value to float
        if isinstance(v, (int, float)):
            value = float(v)
        elif hasattr(v, "item"):  # torch tensor
            value = float(v.item())
        else:
            continue  # Skip non-numeric values

        metric_list.append(Metric(
            key=_sanitize_metric_name(k),
            value=value,
            timestamp=timestamp,
            step=step,
        ))

    if metric_list:
        try:
            client.log_batch(
                run_id=_current_run_id,
                metrics=metric_list,
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")
            # Fallback to individual logging
            for metric in metric_list:
                try:
                    client.log_metric(
                        run_id=_current_run_id,
                        key=metric.key,
                        value=metric.value,
                        step=metric.step,
                    )
                except Exception as e2:
                    logger.warning(f"Failed to log metric {metric.key}: {e2}")


def mlflow_log_param(key: str, value: Any) -> None:
    """
    Log a single parameter using MlflowClient.

    Args:
        key: Parameter name
        value: Parameter value (will be converted to string)
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.log_param(_current_run_id, key, str(value)[:500])
    except Exception as e:
        logger.warning(f"Failed to log param {key}: {e}")


def mlflow_set_tag(key: str, value: str) -> None:
    """
    Set a tag on the current run using MlflowClient.

    Args:
        key: Tag name
        value: Tag value
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.set_tag(_current_run_id, key, value)
    except Exception as e:
        logger.warning(f"Failed to set tag {key}: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file or directory as an artifact using MlflowClient.

    Args:
        local_path: Path to local file or directory
        artifact_path: Optional destination path within artifacts
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.log_artifact(_current_run_id, local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {local_path}: {e}")


def finish_mlflow(status: str = "FINISHED") -> None:
    """
    End the MLflow run using MlflowClient.

    Args:
        status: Run status - "FINISHED", "FAILED", or "KILLED"
    """
    global _current_run_id, _mlflow_client

    if _current_run_id is None:
        return

    client = _get_mlflow_client()

    # Map status string to MLflow RunStatus
    status_map = {
        "FINISHED": "FINISHED",
        "FAILED": "FAILED",
        "KILLED": "KILLED",
    }
    mlflow_status = status_map.get(status.upper(), "FINISHED")

    try:
        client.set_terminated(_current_run_id, status=mlflow_status)
        logger.info(f"MLflow run terminated: {_current_run_id} (status={mlflow_status})")
    except Exception as e:
        logger.warning(f"Failed to terminate run: {e}")

    # Reset global state
    _current_run_id = None


def get_current_run_id() -> Optional[str]:
    """Get the current MLflow run ID."""
    return _current_run_id


def get_mlflow_client() -> Optional["MlflowClient"]:
    """Get the current MlflowClient instance."""
    return _mlflow_client


def reset_mlflow_state() -> None:
    """Reset all MLflow global state. Useful for testing."""
    global _mlflow_client, _current_run_id, _tracking_uri
    _mlflow_client = None
    _current_run_id = None
    _tracking_uri = None

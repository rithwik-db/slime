import logging
import os
from copy import deepcopy

from slime.utils.misc import SingletonMeta

try:
    import mlflow
except ImportError:
    mlflow = None

__all__ = ["init_mlflow_primary", "init_mlflow_secondary", "_MlflowAdapter"]

logger = logging.getLogger(__name__)


def _check_mlflow_available():
    if mlflow is None:
        raise ImportError(
            "mlflow is not installed. Please install it with: pip install mlflow"
        )


def _setup_databricks_tracking():
    """Set up MLflow tracking URI for Databricks if env vars are present."""
    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")

    if databricks_host and databricks_token:
        mlflow.set_tracking_uri("databricks")
        logger.info(f"MLflow tracking URI set to Databricks: {databricks_host}")
        return True
    return False


def init_mlflow_primary(args):
    """Initialize MLflow on the primary process.

    Sets up the MLflow tracking URI (Databricks if env vars present),
    creates/sets the experiment, and starts a new run.
    """
    if not getattr(args, "use_mlflow", False):
        args.mlflow_run_id = None
        return

    _check_mlflow_available()

    # Set tracking URI - prioritize explicit setting, then Databricks env vars
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")
    else:
        if not _setup_databricks_tracking():
            logger.info("MLflow using default local tracking")

    # Set or create experiment
    if args.mlflow_experiment_name:
        mlflow.set_experiment(args.mlflow_experiment_name)
        logger.info(f"MLflow experiment set to: {args.mlflow_experiment_name}")

    # Start the run
    run_name = args.mlflow_run_name
    if run_name and hasattr(args, "rank"):
        run_name = f"{run_name}-RANK_{args.rank}"

    run = mlflow.start_run(run_name=run_name)
    args.mlflow_run_id = run.info.run_id
    logger.info(f"MLflow run started with ID: {args.mlflow_run_id}")

    # Log configuration as parameters
    _log_config_as_params(args)


def _log_config_as_params(args):
    """Log training configuration as MLflow parameters."""
    config = _compute_config_for_logging(args)

    # MLflow has a limit on param value length (500 chars) and total params
    # Filter and truncate as needed
    params_to_log = {}
    for key, value in config.items():
        if value is None:
            continue
        str_value = str(value)
        if len(str_value) > 500:
            str_value = str_value[:497] + "..."
        params_to_log[key] = str_value

    # Log in batches to avoid hitting limits
    batch_size = 100
    param_items = list(params_to_log.items())
    for i in range(0, len(param_items), batch_size):
        batch = dict(param_items[i : i + batch_size])
        try:
            mlflow.log_params(batch)
        except Exception as e:
            logger.warning(f"Failed to log some MLflow params: {e}")


def _compute_config_for_logging(args):
    """Prepare configuration dictionary for logging."""
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        "DATABRICKS_HOST",
    ]
    output["env_vars"] = {
        k: v for k, v in os.environ.items() if k in whitelist_env_vars
    }

    return output


def init_mlflow_secondary(args):
    """Initialize MLflow on secondary processes for distributed training.

    Joins the existing run created by the primary process.
    """
    mlflow_run_id = getattr(args, "mlflow_run_id", None)
    if mlflow_run_id is None:
        return

    _check_mlflow_available()

    # Set tracking URI - same logic as primary
    if getattr(args, "mlflow_tracking_uri", None):
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        _setup_databricks_tracking()

    # Set experiment if specified
    if getattr(args, "mlflow_experiment_name", None):
        mlflow.set_experiment(args.mlflow_experiment_name)

    # Join the existing run
    mlflow.start_run(run_id=mlflow_run_id)
    logger.info(f"MLflow secondary process joined run: {mlflow_run_id}")


class _MlflowAdapter(metaclass=SingletonMeta):
    """Singleton adapter for MLflow metric logging.

    Usage:
        adapter = _MlflowAdapter(args)
        adapter.log({"train/loss": 0.5}, step=100)
    """

    _initialized = False

    def __init__(self, args):
        if not getattr(args, "use_mlflow", False):
            raise ValueError("MLflow is not enabled in args")
        _check_mlflow_available()
        self._initialized = True

    def log(self, data: dict, step: int):
        """Log metrics to MLflow.

        Args:
            data: Dictionary of metric names to values
            step: Current step number
        """
        if not self._initialized:
            return

        # MLflow metric names cannot contain certain characters
        # Replace / with . for compatibility while keeping readability
        sanitized_metrics = {}
        for key, value in data.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
            # Sanitize metric name (MLflow allows alphanumeric, -, _, ., /, and space)
            sanitized_key = key
            sanitized_metrics[sanitized_key] = value

        if sanitized_metrics:
            try:
                mlflow.log_metrics(sanitized_metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log MLflow metrics: {e}")

    def finish(self):
        """End the MLflow run."""
        if self._initialized:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

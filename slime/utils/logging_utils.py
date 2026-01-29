import logging

import wandb

from . import wandb_utils
from . import mlflow_utils
from .tensorboard_utils import _TensorboardAdapter

logger = logging.getLogger(__name__)

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    """
    Initialize all enabled tracking backends.

    Args:
        args: Parsed arguments with logging configuration
        primary: Whether this is the primary process (rank 0 in main controller)
        **kwargs: Additional kwargs passed to init functions
    """
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        mlflow_utils.init_mlflow_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        mlflow_utils.init_mlflow_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    """
    Log metrics to all enabled backends.

    Args:
        args: Parsed arguments with logging configuration
        metrics: Dictionary of metrics to log (must include step_key)
        step_key: Key in metrics dict containing the step number (e.g., "train/step")
    """
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    # MLflow logging
    if getattr(args, "use_mlflow", False):
        step = metrics.get(step_key, 0)
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        mlflow_utils.mlflow_log_metrics(metrics_except_step, step=int(step))


def finish_tracking(args):
    """
    Finalize all tracking backends.

    Call this at the end of training for clean shutdown.
    """
    if getattr(args, "use_mlflow", False):
        mlflow_utils.finish_mlflow()

    # WandB and TensorBoard handle their own cleanup
    logger.info("Tracking finalized")

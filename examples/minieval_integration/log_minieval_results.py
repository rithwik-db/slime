"""Custom eval logging for minieval results.

Usage:
    --custom-eval-rollout-log-function-path examples.minieval_integration.log_minieval_results.log_eval_rollout_data

This avoids the default logger's pass-rate computation which assumes rewards
are grouped by n_samples_per_eval_prompt (minieval produces aggregate scores instead).
"""

import logging

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    """Log minieval eval results to slime's tracking backends.

    Returns True to skip default logging.
    """
    log_dict = {}

    # Log main scores from each eval task
    for task_name, task_data in data.items():
        rewards = task_data["rewards"]
        log_dict[f"eval/{task_name}"] = sum(rewards) / len(rewards)

    # Log detailed minieval metrics
    if extra_metrics:
        log_dict.update(extra_metrics)

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step

    logger.info(f"minieval eval {rollout_id}: {log_dict}")
    logging_utils.log(args, log_dict, step_key="eval/step")

    return True

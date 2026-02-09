"""Custom eval function that delegates evaluation to minieval.

Usage:
    --eval-function-path examples.minieval_integration.eval_with_minieval.generate_rollout

Configure minieval evals inline in slime's YAML config:

    minieval:
      model:
        name: openai         # minieval model type (popped by build_model)
        model: default       # model identifier for API requests (optional, falls back to hf_checkpoint)
        # api_key: EMPTY     # auto-set for local sglang if not specified
        # base_url: ...      # auto-injected from sglang router
      evals:
        - name: gsm8k
        - name: mmlu
      eval_overrides:
        generation_params:
          temperature: 0.0
          max_tokens: 4096
"""

import copy
import logging

from slime.rollout.base_types import RolloutFnEvalOutput

logger = logging.getLogger(__name__)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Run minieval as slime's eval function.

    This function is intended for use with --eval-function-path.
    For training rollouts, use a separate --rollout-function-path.
    """
    if not evaluation:
        raise ValueError(
            "eval_with_minieval is only for evaluation. "
            "Use --rollout-function-path for training rollouts."
        )

    minieval_cfg = getattr(args, "minieval", None)
    if minieval_cfg is None:
        raise ValueError(
            "No 'minieval' section found in config. "
            "Add a root-level 'minieval:' section to your YAML with at least an 'evals' list."
        )

    config = _build_minieval_config(args, minieval_cfg)
    results = _run_minieval(config)
    return _convert_to_slime_output(results)


def _build_minieval_config(args, minieval_cfg):
    """Build a minieval EvalRunnerConfig dict from slime args + inline YAML config.

    minieval's build_model pops "name" from the model dict and passes the rest
    to OpenAIModelConfig, which requires:
      - model: str (model identifier for API requests)
      - base_url: str | None (endpoint URL)
      - api_key: str | None (required by OpenAI client, even for local servers)
    """
    cfg = copy.deepcopy(minieval_cfg)

    # Inject sglang router as the OpenAI-compatible endpoint
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"

    model_cfg = cfg.get("model", {})
    model_cfg.setdefault("name", "openai")
    model_cfg["base_url"] = base_url

    # OpenAIModelConfig requires "model" (the model identifier sent in API requests).
    # Fall back to hf_checkpoint which is what sglang serves the model under.
    if "model" not in model_cfg:
        model_cfg["model"] = getattr(args, "hf_checkpoint", None) or "default"

    # The OpenAI Python client requires a non-empty api_key even for local servers.
    # sglang doesn't validate it, so use a dummy value if none is configured.
    if "api_key" not in model_cfg and "api_key_env" not in model_cfg:
        model_cfg["api_key"] = "EMPTY"

    cfg["model"] = model_cfg

    if "evals" not in cfg or not cfg["evals"]:
        raise ValueError("minieval config must include a non-empty 'evals' list.")

    # loggers, save_folder, save_interval all have safe defaults in EvalRunnerConfig
    cfg.setdefault("loggers", {})

    return cfg


def _run_minieval(config):
    """Invoke minieval's runner and return results."""
    from minieval.utils import builders, config_utils

    runner_config = config_utils.EvalRunnerConfig(**config)
    runner = builders.build_runner_from_config(config=runner_config)

    eval_names = [e.get("name", "unknown") for e in config.get("evals", [])]
    logger.info(f"Running minieval with evals: {eval_names}")

    try:
        results = runner.run()
    except Exception:
        logger.exception("minieval evaluation failed")
        raise

    logger.info(f"minieval completed. Got results for {len(results)} tasks.")
    return results


def _convert_to_slime_output(results):
    """Convert minieval results to RolloutFnEvalOutput.

    minieval returns list[tuple[str, EvalResult]] where EvalResult has:
        .score: float | None
        .metrics: dict[str, float]

    RolloutFnEvalOutput expects:
        data: dict[str, {"rewards": list[float], ...}]
        metrics: dict[str, Any] (optional extra metrics)
    """
    data = {}
    extra_metrics = {}

    for task_name, eval_result in results:
        score = eval_result.score if eval_result.score is not None else 0.0
        data[task_name] = {"rewards": [score], "samples": []}

        if eval_result.metrics:
            for metric_name, metric_value in eval_result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    extra_metrics[f"minieval/{task_name}/{metric_name}"] = metric_value

    return RolloutFnEvalOutput(data=data, metrics=extra_metrics)

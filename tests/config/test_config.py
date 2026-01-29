"""Tests for the Slime configuration module."""

import pytest
import tempfile
from pathlib import Path

# Test imports work
def test_imports():
    """Test that all config module components can be imported."""
    from slime.config.base import (
        SlimeConfig,
        ClusterConfig,
        CheckpointConfig,
        RolloutConfig,
        DataConfig,
        EvalConfig,
        AlgorithmConfig,
        OptimizerConfig,
        RewardConfig,
        SGLangConfig,
        WandbConfig,
        TensorboardConfig,
        MLflowConfig,
        LoggingConfig,
        TrainConfig,
        FSDPConfig,
        MegatronConfig,
        ModelConfig,
        LauncherConfig,
    )
    assert SlimeConfig is not None


class TestSlimeConfig:
    """Tests for SlimeConfig dataclass."""

    def test_default_values(self):
        """Test that SlimeConfig has correct default values."""
        from slime.config.base import SlimeConfig
        config = SlimeConfig()

        assert config.seed == 1234
        assert config.max_gen_len == 8192
        assert config.max_model_len == 10240
        assert config.pretrain_model_name is None

    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        from slime.config.base import SlimeConfig
        config = SlimeConfig()

        # Check cluster defaults
        assert config.cluster.actor_num_nodes == 1
        assert config.cluster.actor_num_gpus_per_node == 8
        assert config.cluster.colocate is False

        # Check optimizer defaults
        assert config.optimizer.lr == 1e-6
        assert config.optimizer.name == "adam"

        # Check algorithm defaults
        assert config.algorithm.advantage_estimator == "grpo"
        assert config.algorithm.eps_clip == 0.2

    def test_custom_values(self):
        """Test creating config with custom values."""
        from slime.config.base import SlimeConfig, ClusterConfig, OptimizerConfig

        config = SlimeConfig(
            seed=42,
            cluster=ClusterConfig(actor_num_nodes=2, rollout_num_gpus=4),
            optimizer=OptimizerConfig(lr=1e-5, name="adamw"),
        )

        assert config.seed == 42
        assert config.cluster.actor_num_nodes == 2
        assert config.cluster.rollout_num_gpus == 4
        assert config.optimizer.lr == 1e-5
        assert config.optimizer.name == "adamw"


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_config_defaults(self):
        """Test loading config with no file gives defaults."""
        from slime.config.loader import load_config

        config = load_config()
        assert config.seed == 1234
        assert config.optimizer.lr == 1e-6

    def test_load_config_from_yaml(self):
        """Test loading config from YAML file."""
        from slime.config.loader import load_config

        yaml_content = """
seed: 42
optimizer:
  lr: 1e-5
  name: adamw
cluster:
  actor_num_nodes: 2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            assert config.seed == 42
            assert config.optimizer.lr == 1e-5
            assert config.optimizer.name == "adamw"
            assert config.cluster.actor_num_nodes == 2

        Path(f.name).unlink()

    def test_load_config_with_parameters_key(self):
        """Test loading config with 'parameters:' wrapper."""
        from slime.config.loader import load_config

        yaml_content = """
parameters:
  seed: 42
  optimizer:
    lr: 1e-5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            assert config.seed == 42
            assert config.optimizer.lr == 1e-5

        Path(f.name).unlink()

    def test_load_config_with_cli_overrides(self):
        """Test CLI overrides take precedence."""
        from slime.config.loader import load_config

        yaml_content = """
seed: 42
optimizer:
  lr: 1e-5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(
                f.name,
                cli_overrides=["seed=100", "optimizer.lr=1e-7"]
            )

            assert config.seed == 100
            assert config.optimizer.lr == 1e-7

        Path(f.name).unlink()

    def test_load_config_variable_interpolation(self):
        """Test OmegaConf variable interpolation."""
        from slime.config.loader import load_config

        yaml_content = """
max_gen_len: 8192
max_model_len: 10240
rollout:
  max_response_len: ${max_gen_len}
  max_context_len: ${max_model_len}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name, resolve=True)

            assert config.rollout.max_response_len == 8192
            assert config.rollout.max_context_len == 10240

        Path(f.name).unlink()

    def test_load_config_file_not_found(self):
        """Test FileNotFoundError for missing config."""
        from slime.config.loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestConfigConverter:
    """Tests for config conversion utilities."""

    def test_config_to_namespace(self):
        """Test converting config to argparse Namespace."""
        from slime.config.loader import load_config
        from slime.config.converter import config_to_namespace

        config = load_config()
        ns = config_to_namespace(config)

        assert hasattr(ns, 'seed')
        assert ns.seed == 1234

    def test_flatten_config(self):
        """Test _flatten_config function."""
        from slime.config.loader import _flatten_config

        nested = {
            'optimizer': {
                'lr': 1e-6,
                'name': 'adam'
            },
            'seed': 42
        }

        flat = _flatten_config(nested)

        assert 'optimizer_lr' in flat
        assert 'optimizer_name' in flat
        assert 'seed' in flat
        assert flat['optimizer_lr'] == 1e-6
        assert flat['seed'] == 42


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        from slime.config.loader import load_config
        from slime.config.validation import validate_config

        config = load_config()
        validated = validate_config(config)

        # Derived defaults should be applied
        assert validated.algorithm.eps_clip_high == validated.algorithm.eps_clip
        assert validated.cluster.critic_num_nodes == validated.cluster.actor_num_nodes

    def test_validate_invalid_advantage_estimator(self):
        """Test validation fails for invalid advantage estimator."""
        from slime.config.loader import load_config
        from slime.config.validation import validate_config, ConfigValidationError

        config = load_config(cli_overrides=["algorithm.advantage_estimator=invalid"])

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "advantage_estimator" in str(exc_info.value)

    def test_validate_invalid_backend(self):
        """Test validation fails for invalid train backend."""
        from slime.config.loader import load_config
        from slime.config.validation import validate_config, ConfigValidationError

        config = load_config(cli_overrides=["train.backend=invalid"])

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "train_backend" in str(exc_info.value)

    def test_validate_dynamic_batch_size_requires_max_tokens(self):
        """Test validation fails when dynamic batch size is enabled without max_tokens_per_gpu."""
        from slime.config.loader import load_config
        from slime.config.validation import validate_config, ConfigValidationError

        config = load_config(cli_overrides=["data.use_dynamic_batch_size=true"])

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)

        assert "max_tokens_per_gpu" in str(exc_info.value)


class TestYamlAsDict:
    """Tests for load_yaml_as_dict function."""

    def test_load_yaml_as_dict(self):
        """Test loading YAML and flattening to dict."""
        from slime.config.loader import load_yaml_as_dict

        yaml_content = """
seed: 42
optimizer:
  lr: 1e-5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            flat = load_yaml_as_dict(f.name)

            assert flat['seed'] == 42
            assert flat['optimizer_lr'] == 1e-5
            # Shorthand keys should also exist
            assert flat['lr'] == 1e-5

        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

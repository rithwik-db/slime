"""Integration tests for arguments.py YAML config loading."""

import pytest
import sys
import tempfile
from pathlib import Path


class TestArgumentsConfigIntegration:
    """Tests for YAML config integration with arguments.py."""

    def test_extract_config_path_from_argv(self):
        """Test extracting --config path from command line."""
        from slime.utils.arguments import _extract_config_path_from_argv

        # Test --config value format
        assert _extract_config_path_from_argv(["--config", "/path/to/config.yaml"]) == "/path/to/config.yaml"

        # Test --config=value format
        assert _extract_config_path_from_argv(["--config=/path/to/config.yaml"]) == "/path/to/config.yaml"

        # Test with other args
        assert _extract_config_path_from_argv(
            ["--seed", "42", "--config", "my_config.yaml", "--lr", "1e-5"]
        ) == "my_config.yaml"

        # Test no config
        assert _extract_config_path_from_argv(["--seed", "42"]) is None

        # Test empty
        assert _extract_config_path_from_argv([]) is None

    def test_extract_explicit_args(self):
        """Test extracting explicit CLI arguments."""
        from slime.utils.arguments import _extract_explicit_args

        explicit = _extract_explicit_args(["--seed", "42", "--config", "test.yaml"])
        assert "seed" in explicit
        assert "config" in explicit

        # Test --key=value format
        explicit = _extract_explicit_args(["--learning-rate=1e-5"])
        assert "learning_rate" in explicit

        # Test hyphen to underscore conversion
        explicit = _extract_explicit_args(["--actor-num-nodes", "2"])
        assert "actor_num_nodes" in explicit

    def test_load_yaml_defaults(self):
        """Test loading YAML defaults."""
        from slime.utils.arguments import _load_yaml_defaults

        yaml_content = """
seed: 42
optimizer:
  lr: 1e-5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            defaults = _load_yaml_defaults(f.name)

            assert defaults['seed'] == 42
            assert defaults['optimizer_lr'] == 1e-5
            # Shorthand key should exist
            assert defaults.get('lr') == 1e-5

        Path(f.name).unlink()

    def test_load_yaml_defaults_with_parameters_wrapper(self):
        """Test loading YAML with parameters: wrapper."""
        from slime.utils.arguments import _load_yaml_defaults

        yaml_content = """
parameters:
  seed: 123
  max_gen_len: 4096
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            defaults = _load_yaml_defaults(f.name)

            assert defaults['seed'] == 123
            assert defaults['max_gen_len'] == 4096

        Path(f.name).unlink()

    def test_yaml_defaults_applied_to_namespace(self):
        """Test that YAML defaults are applied to namespace."""
        import argparse
        from slime.config.converter import apply_yaml_defaults_to_namespace

        # Create a namespace with some defaults and Nones
        args = argparse.Namespace(
            seed=1234,  # Default value
            lr=None,  # Not set
            batch_size=32,  # Default value
        )

        yaml_config = {
            'seed': 42,
            'lr': 1e-5,
            'batch_size': 64,
        }

        # Only 'seed' was explicitly set
        explicit = {'seed'}

        result = apply_yaml_defaults_to_namespace(args, yaml_config, explicit)

        # seed should NOT be overwritten (was explicit)
        assert result.seed == 1234
        # lr should be set from YAML (was None)
        assert result.lr == 1e-5
        # batch_size should NOT be set (was not None, even though not explicit)
        assert result.batch_size == 32


class TestConfigVariableInterpolation:
    """Test OmegaConf variable interpolation through arguments."""

    def test_variable_interpolation_in_yaml(self):
        """Test that ${var} interpolation works in YAML configs."""
        from slime.utils.arguments import _load_yaml_defaults

        yaml_content = """
parameters:
  pretrain_model_name: /models/test-model
  max_gen_len: 8192
  checkpoint:
    hf_checkpoint: ${pretrain_model_name}
  rollout:
    max_response_len: ${max_gen_len}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            defaults = _load_yaml_defaults(f.name)

            # Check interpolation worked
            assert defaults.get('checkpoint_hf_checkpoint') == '/models/test-model'
            assert defaults.get('rollout_max_response_len') == 8192

        Path(f.name).unlink()


class TestExampleConfigLoading:
    """Test loading the example config file."""

    def test_load_example_qwen_config(self):
        """Test loading the example qwen3-4b-grpo.yaml config."""
        from slime.config.loader import load_config

        config_path = Path("/root/slime/configs/training/qwen3-4b-grpo.yaml")
        if not config_path.exists():
            pytest.skip("Example config not found")

        config = load_config(str(config_path))

        # Check top-level values
        assert config.seed == 1234
        assert config.max_gen_len == 8192

        # Check nested values
        assert config.cluster.actor_num_nodes == 1
        assert config.optimizer.lr == 1e-6
        assert config.algorithm.advantage_estimator == "grpo"

        # Check interpolation worked
        assert config.rollout.max_response_len == 8192  # ${max_gen_len}
        assert config.rollout.max_context_len == 10240  # ${max_model_len}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

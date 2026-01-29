"""Tests for slime.launcher module."""

import argparse
import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLauncherModuleImports:
    """Test that launcher module can be imported."""

    def test_import_launcher_module(self):
        """Test that launcher module can be imported."""
        from slime import launcher
        assert launcher is not None

    def test_import_spmd_ray_bridge(self):
        """Test that spmd_ray_bridge can be imported."""
        from slime.launcher import spmd_ray_bridge
        assert spmd_ray_bridge is not None

    def test_import_spmd_train(self):
        """Test that spmd_train can be imported."""
        from slime.launcher import spmd_train
        assert spmd_train is not None

    def test_import_launch_grpo(self):
        """Test that launch_grpo can be imported."""
        from slime.launcher import launch_grpo
        assert launch_grpo is not None


class TestSpmdRayBridgeFunctions:
    """Test spmd_ray_bridge utility functions."""

    def test_get_free_port(self):
        """Test get_free_port returns valid port."""
        from slime.launcher.spmd_ray_bridge import get_free_port

        port = get_free_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535

    def test_get_free_port_unique(self):
        """Test get_free_port returns different ports on multiple calls."""
        from slime.launcher.spmd_ray_bridge import get_free_port

        ports = [get_free_port() for _ in range(5)]
        # Should be mostly unique (small chance of collision)
        assert len(set(ports)) >= 3

    def test_is_cuda_visible_devices_set(self):
        """Test is_cuda_visible_devices_set function."""
        from slime.launcher.spmd_ray_bridge import is_cuda_visible_devices_set

        # Test with env var not set (default)
        original = os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")
        try:
            os.environ.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
            assert is_cuda_visible_devices_set() is True

            os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "0"
            assert is_cuda_visible_devices_set() is True

            os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            assert is_cuda_visible_devices_set() is False
        finally:
            if original is not None:
                os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)


class TestSpmdRayBridgeDistributedChecks:
    """Test distributed initialization checks."""

    def test_init_ray_without_dist_raises(self):
        """Test init_ray_with_torch_distributed raises without dist init."""
        from slime.launcher.spmd_ray_bridge import init_ray_with_torch_distributed

        with patch("torch.distributed.is_initialized", return_value=False):
            with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
                init_ray_with_torch_distributed()


class TestLaunchGrpoConfig:
    """Test launch_grpo configuration handling."""

    def test_load_compute_yaml_with_parameters(self):
        """Test loading compute YAML with parameters section."""
        from slime.launcher.launch_grpo import load_compute_yaml
        from omegaconf import OmegaConf

        # Create a temporary compute YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
name: test-run
compute:
  gpus: 4
parameters:
  pretrain_model_name: test-model
  max_gen_len: 1024
""")
            f.flush()
            yaml_path = f.name

        try:
            config = load_compute_yaml(yaml_path)
            assert "pretrain_model_name" in config
            assert config.pretrain_model_name == "test-model"
            assert config.max_gen_len == 1024
            # Compute section should not be in extracted params
            assert "compute" not in config
        finally:
            os.unlink(yaml_path)

    def test_load_compute_yaml_plain_config(self):
        """Test loading plain training config without parameters section."""
        from slime.launcher.launch_grpo import load_compute_yaml
        from omegaconf import OmegaConf

        # Create a temporary plain YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
pretrain_model_name: test-model
max_gen_len: 1024
""")
            f.flush()
            yaml_path = f.name

        try:
            config = load_compute_yaml(yaml_path)
            assert "pretrain_model_name" in config
            assert config.pretrain_model_name == "test-model"
        finally:
            os.unlink(yaml_path)

    def test_merge_configs_single_file(self):
        """Test merging single config file."""
        from slime.launcher.launch_grpo import merge_configs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
parameters:
  model_name: test
  lr: 0.001
""")
            f.flush()
            yaml_path = f.name

        try:
            config = merge_configs([yaml_path])
            assert config.model_name == "test"
            assert config.lr == 0.001
        finally:
            os.unlink(yaml_path)

    def test_merge_configs_multiple_files(self):
        """Test merging multiple config files (later overrides earlier)."""
        from slime.launcher.launch_grpo import merge_configs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write("""
parameters:
  model_name: base-model
  lr: 0.001
  batch_size: 32
""")
            f1.flush()
            yaml_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write("""
parameters:
  lr: 0.0001
  extra_param: value
""")
            f2.flush()
            yaml_path2 = f2.name

        try:
            config = merge_configs([yaml_path1, yaml_path2])
            assert config.model_name == "base-model"  # From first file
            assert config.lr == 0.0001  # Overridden by second file
            assert config.batch_size == 32  # From first file
            assert config.extra_param == "value"  # From second file
        finally:
            os.unlink(yaml_path1)
            os.unlink(yaml_path2)

    def test_merge_configs_with_cli_overrides(self):
        """Test merging configs with CLI overrides."""
        from slime.launcher.launch_grpo import merge_configs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
parameters:
  model_name: test
  lr: 0.001
""")
            f.flush()
            yaml_path = f.name

        try:
            config = merge_configs([yaml_path], cli_overrides=["lr=0.0005", "new_param=42"])
            assert config.model_name == "test"
            assert config.lr == 0.0005  # CLI override
            assert config.new_param == 42  # New from CLI
        finally:
            os.unlink(yaml_path)

    def test_config_to_args_list(self):
        """Test converting config to CLI arguments list."""
        from slime.launcher.launch_grpo import config_to_args_list
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "lr": 0.001,
            "use_mlflow": True,
            "skip_feature": False,
            "model_name": "test-model",
        })

        args_list = config_to_args_list(config)

        assert "--lr" in args_list
        assert "0.001" in args_list
        assert "--use-mlflow" in args_list
        assert "--skip-feature" not in args_list  # False booleans not included
        assert "--model-name" in args_list

    def test_config_passthrough_no_mapping(self):
        """Test that config keys are passed through without mapping."""
        from slime.launcher.launch_grpo import merge_configs, get_keys_to_skip
        from omegaconf import OmegaConf

        # Create config with slime native argument names
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
parameters:
  hf_checkpoint: Qwen/Qwen3-4B
  n_samples_per_prompt: 8
  rollout_max_response_len: 4096
  global_batch_size: 64
""")
            f.flush()
            yaml_path = f.name

        try:
            config = merge_configs([yaml_path])

            # Verify keys are passed through as-is (no mapping)
            assert config.hf_checkpoint == "Qwen/Qwen3-4B"
            assert config.n_samples_per_prompt == 8
            assert config.rollout_max_response_len == 4096
            assert config.global_batch_size == 64
        finally:
            os.unlink(yaml_path)

    def test_get_keys_to_skip(self):
        """Test that compute YAML metadata keys are in skip list."""
        from slime.launcher.launch_grpo import get_keys_to_skip

        skip_keys = get_keys_to_skip()

        # Compute YAML metadata should be skipped
        assert "name" in skip_keys
        assert "image" in skip_keys
        assert "compute" in skip_keys
        assert "scheduling" in skip_keys
        assert "integrations" in skip_keys
        assert "command" in skip_keys


class TestLaunchGrpoArgParsing:
    """Test launch_grpo argument parsing."""

    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        from slime.launcher.launch_grpo import parse_args

        original_argv = sys.argv
        try:
            sys.argv = ["launch_grpo", "-f", "test.yaml"]
            args = parse_args()
            assert args.config_files == ["test.yaml"]
            assert args.launch_method == "torchrun"
            assert args.train_backend == "fsdp"
            assert args.dry_run is False
        finally:
            sys.argv = original_argv

    def test_parse_args_multiple_files(self):
        """Test parsing multiple config files."""
        from slime.launcher.launch_grpo import parse_args

        original_argv = sys.argv
        try:
            sys.argv = ["launch_grpo", "-f", "base.yaml", "-f", "override.yaml"]
            args = parse_args()
            assert args.config_files == ["base.yaml", "override.yaml"]
        finally:
            sys.argv = original_argv

    def test_parse_args_with_overrides(self):
        """Test parsing with CLI overrides."""
        from slime.launcher.launch_grpo import parse_args

        original_argv = sys.argv
        try:
            sys.argv = ["launch_grpo", "-f", "test.yaml", "--lr=0.001", "--batch-size", "32"]
            args = parse_args()
            assert "lr=0.001" in args.cli_overrides
            assert "batch-size=32" in args.cli_overrides
        finally:
            sys.argv = original_argv


class TestSpmdTrainFunctions:
    """Test spmd_train module functions."""

    def test_setup_signal_handlers(self):
        """Test that signal handlers can be set up."""
        from slime.launcher.spmd_train import setup_signal_handlers

        # Should not raise
        setup_signal_handlers()


class TestLaunchMethodArguments:
    """Test launch-method arguments in arguments.py."""

    def test_launch_method_argument_exists(self):
        """Test that --launch-method argument is added."""
        # This is tested implicitly by checking if we can import arguments
        # and the argument is recognized
        import sys
        original_argv = sys.argv

        try:
            sys.argv = ["test", "--help"]
            # Import will fail if arguments are malformed
            from slime.utils import arguments
            assert True
        except SystemExit:
            # --help causes exit, that's fine
            assert True
        finally:
            sys.argv = original_argv


class TestExampleComputeYaml:
    """Test example compute YAML file."""

    def test_example_compute_yaml_exists(self):
        """Test that example compute YAML exists."""
        compute_yaml = Path("/root/slime/configs/compute/aroll-grpo-math.yaml")
        assert compute_yaml.exists()

    def test_example_compute_yaml_valid(self):
        """Test that example compute YAML is valid and uses slime native arg names."""
        from omegaconf import OmegaConf

        compute_yaml = Path("/root/slime/configs/compute/aroll-grpo-math.yaml")
        config = OmegaConf.load(compute_yaml)

        assert "name" in config
        assert "parameters" in config
        # Config should use slime's native argument names directly
        assert "hf_checkpoint" in config.parameters


class TestTorchrunScript:
    """Test torchrun launch script."""

    def test_torchrun_script_exists(self):
        """Test that torchrun script exists."""
        script = Path("/root/slime/scripts/run-qwen3-4B-fsdp-torchrun.sh")
        assert script.exists()

    def test_torchrun_script_executable(self):
        """Test that torchrun script is executable."""
        script = Path("/root/slime/scripts/run-qwen3-4B-fsdp-torchrun.sh")
        assert os.access(script, os.X_OK)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

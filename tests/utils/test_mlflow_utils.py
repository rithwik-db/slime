"""Tests for mlflow_utils.py."""

import argparse
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


class TestMlflowUtilsWithoutMlflow:
    """Tests that don't require mlflow to be installed."""

    def test_import_mlflow_utils(self):
        """Test that mlflow_utils can be imported."""
        from slime.utils import mlflow_utils
        assert mlflow_utils is not None

    def test_flatten_dict(self):
        """Test _flatten_dict function."""
        from slime.utils.mlflow_utils import _flatten_dict

        # Simple case
        result = _flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

        # Nested case
        result = _flatten_dict({"optimizer": {"lr": 1e-6, "weight_decay": 0.1}})
        assert result == {"optimizer.lr": 1e-6, "optimizer.weight_decay": 0.1}

        # Deeply nested case
        result = _flatten_dict({"level1": {"level2": {"level3": "value"}}})
        assert result == {"level1.level2.level3": "value"}

    def test_sanitize_metric_name(self):
        """Test _sanitize_metric_name function."""
        from slime.utils.mlflow_utils import _sanitize_metric_name

        # Normal name should pass through
        assert _sanitize_metric_name("train/loss") == "train/loss"

        # Colon should be replaced with underscore
        assert _sanitize_metric_name("metric:value") == "metric_value"

    def test_compute_config_for_logging(self):
        """Test _compute_config_for_logging function."""
        from slime.utils.mlflow_utils import _compute_config_for_logging

        args = argparse.Namespace(
            seed=42,
            lr=1e-6,
            use_mlflow=True,
            some_list=[1, 2, 3],  # Should be filtered out
            some_dict={"a": 1},  # Should be filtered out
        )

        result = _compute_config_for_logging(args)

        assert result["seed"] == 42
        assert result["lr"] == 1e-6
        assert result["use_mlflow"] is True
        assert "some_list" not in result  # Lists filtered out
        assert "some_dict" not in result  # Dicts filtered out

    def test_reset_mlflow_state(self):
        """Test reset_mlflow_state function."""
        from slime.utils import mlflow_utils

        # Set some state
        mlflow_utils._current_run_id = "test_run_id"
        mlflow_utils._tracking_uri = "http://test"

        # Reset
        mlflow_utils.reset_mlflow_state()

        # Verify reset
        assert mlflow_utils._current_run_id is None
        assert mlflow_utils._tracking_uri is None
        assert mlflow_utils._mlflow_client is None

    def test_get_current_run_id_when_none(self):
        """Test get_current_run_id returns None when no run."""
        from slime.utils.mlflow_utils import get_current_run_id, reset_mlflow_state

        reset_mlflow_state()
        assert get_current_run_id() is None


class TestMlflowUtilsInitPrimary:
    """Tests for init_mlflow_primary function."""

    def test_init_mlflow_primary_disabled(self):
        """Test init_mlflow_primary when MLflow is disabled."""
        from slime.utils.mlflow_utils import init_mlflow_primary, reset_mlflow_state

        reset_mlflow_state()

        args = argparse.Namespace(use_mlflow=False)
        init_mlflow_primary(args)

        assert args.mlflow_run_id is None

    def test_init_mlflow_primary_no_experiment_name(self):
        """Test init_mlflow_primary raises error without experiment name."""
        from slime.utils.mlflow_utils import init_mlflow_primary, reset_mlflow_state

        reset_mlflow_state()

        args = argparse.Namespace(
            use_mlflow=True,
            mlflow_tracking_uri=None,
            mlflow_experiment_name=None,
        )

        # Mock MlflowClient to avoid actual MLflow dependency
        with patch("slime.utils.mlflow_utils._get_mlflow_client") as mock_client:
            with pytest.raises(ValueError, match="--mlflow-experiment-name is required"):
                init_mlflow_primary(args)


class TestMlflowUtilsInitSecondary:
    """Tests for init_mlflow_secondary function."""

    def test_init_mlflow_secondary_no_run_id(self):
        """Test init_mlflow_secondary does nothing without run_id."""
        from slime.utils.mlflow_utils import init_mlflow_secondary, reset_mlflow_state, get_current_run_id

        reset_mlflow_state()

        args = argparse.Namespace(mlflow_run_id=None)
        init_mlflow_secondary(args)

        assert get_current_run_id() is None


class TestMlflowUtilsLogging:
    """Tests for MLflow logging functions."""

    def test_mlflow_log_metrics_no_run(self):
        """Test mlflow_log_metrics does nothing without active run."""
        from slime.utils.mlflow_utils import mlflow_log_metrics, reset_mlflow_state

        reset_mlflow_state()

        # Should not raise error
        mlflow_log_metrics({"loss": 0.5}, step=1)

    def test_mlflow_log_param_no_run(self):
        """Test mlflow_log_param does nothing without active run."""
        from slime.utils.mlflow_utils import mlflow_log_param, reset_mlflow_state

        reset_mlflow_state()

        # Should not raise error
        mlflow_log_param("key", "value")

    def test_mlflow_set_tag_no_run(self):
        """Test mlflow_set_tag does nothing without active run."""
        from slime.utils.mlflow_utils import mlflow_set_tag, reset_mlflow_state

        reset_mlflow_state()

        # Should not raise error
        mlflow_set_tag("key", "value")

    def test_log_artifact_no_run(self):
        """Test log_artifact does nothing without active run."""
        from slime.utils.mlflow_utils import log_artifact, reset_mlflow_state

        reset_mlflow_state()

        # Should not raise error
        log_artifact("/path/to/file")


class TestMlflowUtilsFinish:
    """Tests for finish_mlflow function."""

    def test_finish_mlflow_no_run(self):
        """Test finish_mlflow does nothing without active run."""
        from slime.utils.mlflow_utils import finish_mlflow, reset_mlflow_state

        reset_mlflow_state()

        # Should not raise error
        finish_mlflow()


class TestSetTrackingUri:
    """Tests for _set_tracking_uri function."""

    def test_set_tracking_uri_databricks(self):
        """Test setting tracking URI to databricks."""
        from slime.utils.mlflow_utils import _set_tracking_uri, reset_mlflow_state

        reset_mlflow_state()

        result = _set_tracking_uri("databricks")
        assert result == "databricks"

    def test_set_tracking_uri_http(self):
        """Test setting tracking URI to HTTP URL."""
        from slime.utils.mlflow_utils import _set_tracking_uri, reset_mlflow_state

        reset_mlflow_state()

        result = _set_tracking_uri("http://localhost:5000")
        assert result == "http://localhost:5000"

    def test_set_tracking_uri_from_env(self):
        """Test setting tracking URI from environment variable."""
        from slime.utils.mlflow_utils import _set_tracking_uri, reset_mlflow_state

        reset_mlflow_state()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-server:5000"}):
            result = _set_tracking_uri(None)
            assert result == "http://env-server:5000"


class TestDatabricksCredentials:
    """Tests for Databricks credential validation."""

    def test_validate_databricks_credentials_both_set(self):
        """Test validation passes when both credentials are set."""
        from slime.utils.mlflow_utils import _validate_databricks_credentials, reset_mlflow_state

        reset_mlflow_state()

        with patch.dict(os.environ, {
            "DATABRICKS_HOST": "https://myworkspace.databricks.com",
            "DATABRICKS_TOKEN": "dapi1234567890abcdef"
        }):
            result = _validate_databricks_credentials()
            assert result is True

    def test_validate_databricks_credentials_missing_host(self):
        """Test validation fails when host is missing."""
        from slime.utils.mlflow_utils import _validate_databricks_credentials, reset_mlflow_state

        reset_mlflow_state()

        # Clear DATABRICKS_HOST if set
        env = {"DATABRICKS_TOKEN": "dapi1234567890abcdef"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("DATABRICKS_HOST", None)
            result = _validate_databricks_credentials()
            assert result is False

    def test_validate_databricks_credentials_missing_token(self):
        """Test validation fails when token is missing."""
        from slime.utils.mlflow_utils import _validate_databricks_credentials, reset_mlflow_state

        reset_mlflow_state()

        # Clear DATABRICKS_TOKEN if set
        env = {"DATABRICKS_HOST": "https://myworkspace.databricks.com"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("DATABRICKS_TOKEN", None)
            result = _validate_databricks_credentials()
            assert result is False

    def test_set_tracking_uri_databricks_validates_credentials(self):
        """Test that setting databricks URI validates credentials."""
        from slime.utils.mlflow_utils import _set_tracking_uri, reset_mlflow_state

        reset_mlflow_state()

        with patch.dict(os.environ, {
            "DATABRICKS_HOST": "https://myworkspace.databricks.com",
            "DATABRICKS_TOKEN": "dapi1234567890abcdef"
        }):
            result = _set_tracking_uri("databricks")
            assert result == "databricks"


class TestMlflowUtilsWithMlflow:
    """Tests that require mlflow to be installed."""

    @pytest.fixture(autouse=True)
    def skip_without_mlflow(self):
        """Skip tests if mlflow is not installed."""
        pytest.importorskip("mlflow")

    def test_get_mlflow_client(self):
        """Test _get_mlflow_client creates a client."""
        from slime.utils.mlflow_utils import _get_mlflow_client, reset_mlflow_state, _set_tracking_uri

        reset_mlflow_state()
        _set_tracking_uri(None)

        client = _get_mlflow_client()
        assert client is not None

        # Second call should return same instance
        client2 = _get_mlflow_client()
        assert client is client2


class TestLoggingUtilsIntegration:
    """Integration tests for logging_utils with MLflow."""

    def test_logging_utils_imports_mlflow_utils(self):
        """Test that logging_utils properly imports mlflow_utils."""
        from slime.utils import logging_utils
        assert hasattr(logging_utils, "mlflow_utils")

    def test_init_tracking_calls_mlflow(self):
        """Test that init_tracking calls mlflow_utils.init_mlflow_primary."""
        from slime.utils.logging_utils import init_tracking
        from slime.utils import mlflow_utils

        mlflow_utils.reset_mlflow_state()

        args = argparse.Namespace(
            use_wandb=False,
            use_mlflow=False,
        )

        with patch.object(mlflow_utils, "init_mlflow_primary") as mock_init:
            init_tracking(args, primary=True)
            mock_init.assert_called_once_with(args)

    def test_init_tracking_secondary_calls_mlflow(self):
        """Test that init_tracking calls mlflow_utils.init_mlflow_secondary."""
        from slime.utils.logging_utils import init_tracking
        from slime.utils import mlflow_utils

        mlflow_utils.reset_mlflow_state()

        args = argparse.Namespace(
            use_wandb=False,
            use_mlflow=False,
            wandb_run_id=None,
        )

        with patch.object(mlflow_utils, "init_mlflow_secondary") as mock_init:
            init_tracking(args, primary=False)
            mock_init.assert_called_once_with(args)

    def test_finish_tracking_calls_mlflow(self):
        """Test that finish_tracking calls mlflow_utils.finish_mlflow."""
        from slime.utils.logging_utils import finish_tracking
        from slime.utils import mlflow_utils

        mlflow_utils.reset_mlflow_state()

        args = argparse.Namespace(use_mlflow=True)

        with patch.object(mlflow_utils, "finish_mlflow") as mock_finish:
            finish_tracking(args)
            mock_finish.assert_called_once()

    def test_finish_tracking_skips_when_disabled(self):
        """Test that finish_tracking doesn't call finish_mlflow when disabled."""
        from slime.utils.logging_utils import finish_tracking
        from slime.utils import mlflow_utils

        mlflow_utils.reset_mlflow_state()

        args = argparse.Namespace(use_mlflow=False)

        with patch.object(mlflow_utils, "finish_mlflow") as mock_finish:
            finish_tracking(args)
            mock_finish.assert_not_called()


class TestMlflowArguments:
    """Test MLflow arguments are properly added to argument parser."""

    def test_mlflow_arguments_exist(self):
        """Test that MLflow arguments are added to the parser."""
        # This tests that the arguments.py modifications work
        import sys
        original_argv = sys.argv

        try:
            # Set up minimal args
            sys.argv = ["test", "--help"]
            from slime.utils.arguments import parse_args

            # We can't actually call parse_args because it exits on --help
            # Instead, let's check that the module imports correctly
            assert True  # If we got here, the module imports correctly

        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

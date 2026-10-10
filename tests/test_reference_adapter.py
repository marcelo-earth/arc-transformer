from argparse import Namespace
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from reference_runner import logged_builder, checkpoint_for_evaluation


class AdapterTests(unittest.TestCase):
    def test_evaluation_kwargs_and_config_are_preserved(self):
        cfg = Namespace(epochs=90, train_log_mode="never", log_location="none")
        before = deepcopy(vars(cfg))
        checkpoint = {"model_state": "frozen"}
        original = Mock(return_value="built evaluation model")
        result = logged_builder(original, probe=True)(cfg, checkpoint=checkpoint, is_eval=True)
        self.assertEqual(result, "built evaluation model")
        self.assertEqual(vars(cfg), before)
        original.assert_called_once_with(cfg, checkpoint=checkpoint, is_eval=True)

    def test_high_preserves_reference_checkpoints_and_hyperparameters(self):
        cfg = Namespace(epochs=650, max_augments=300, seed=42,
                        checkpoint_epochs=[645, 648, 650])
        original = Mock()
        with patch("reference_runner.Path.write_text"):
            logged_builder(original)(cfg)
        self.assertEqual(cfg.checkpoint_epochs, [100, 200, 300, 400, 500, 600, 645, 648, 650])
        self.assertEqual((cfg.epochs, cfg.max_augments, cfg.seed), (650, 300, 42))
        original.assert_called_once_with(cfg)

    def test_high_recovery_selects_648_instead_of_final_checkpoint(self):
        self.assertEqual(checkpoint_for_evaluation("/runs/high/artifacts", Namespace(epochs=650)),
                         Path("/runs/high/artifacts/tiny.epoch648.pt"))
        self.assertEqual(checkpoint_for_evaluation("/runs/low/artifacts", Namespace(epochs=90)),
                         Path("/runs/low/artifacts/tiny.pt"))

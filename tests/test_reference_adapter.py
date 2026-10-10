from argparse import Namespace
from copy import deepcopy
import unittest
from unittest.mock import Mock

from reference_runner import logged_builder


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

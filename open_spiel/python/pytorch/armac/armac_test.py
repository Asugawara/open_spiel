from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized

from torch.testing._internal.common_utils import run_tests, TestCase
import torch
import torch.nn as nn

import pyspiel

from armac import armac

class ARMACTest(parameterized.TestCase, TestCase):
  @parameterized.parameters("kuhn_poker", "leduc_poker")
  def test_armac(self, game):
      armac(game, 10, 64, 5, 2)


if __name__=="__main__":
  run_tests()
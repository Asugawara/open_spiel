# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Double Neural Counterfactual Regret Minimization (DNCFR) [Hui Li et al., 2018].
a double neural representation for the imperfect information games, where one neural network represents
the cumulative regret, and the other represents the average strategy. Furthermore, we adopt the counterfactual 
regret minimization algorithm to optimize this double neural representation. To make neural learning efficient, we
also developed several novel techniques including a robust sampling method, mini-batch Monte
Carlo Counterfactual Regret Minimization (MCCFR) and Monte Carlo Counterfactual Regret
Minimization Plus (MCCFR+) which may be of independent interests. Experimentally, we
demonstrate that the proposed double neural algorithm converges significantly better than the
reinforcement learning counterpart.


# References
Hui Li, Kailiang Hu, Zhibang Ge, Tao Jiang, Yuan Qi, and Le Song. 
    Double Neural Counterfactual Regret Minimization.
    At the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import torch
import torch.nn as nn
import torch.nn.functional as F



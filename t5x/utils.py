# coding=utf-8
# Copyright 2022 The Optformer Neurips2022 Authors.
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

"""Utility functions and types."""
import enum
from typing import Tuple, Union

import tensorflow as tf

Int = Union[int, tf.Tensor]


class TargetType(enum.Enum):
  SEPARATOR = 1
  PARAMETER = 2
  FUNCTION = 3
  OTHER = 4


class TrialTokenScheme(object):
  """Calculate trial token allocations.

  Every trial is in the format of "[parameters] * fun_value |".

  The entire token sequence is in the format of
    "INIT_TOKENS TRIAL_0 TRIAL_1 ... TRIAL_LAST_EXCEPT_|"
  """

  def __init__(self, num_initial_tokens):
    self.num_initial_tokens = num_initial_tokens

  def trial_length(self, num_parameters):
    return num_parameters + 3

  def max_trials(self, num_parameters, sequence_length):
    """Compute the maximum number of trials in a sequence."""
    num_trials = (sequence_length - self.num_initial_tokens
                  + 1  # The "|" token of the last trial is not needed.
                  ) // self.trial_length(num_parameters)
    return num_trials

  def fun_index_in_trial(self, num_parameters, trial_index):
    """Index of the function token in a trial."""
    trial_len = self.trial_length(num_parameters)
    return (self.num_initial_tokens + (trial_index + 1) * trial_len - 1) - 1

  def param_index_range_in_trial(self, num_parameters,
                                 trial_index):
    """Start and end (excluded) indices of the parameter tokens in a trial."""
    trial_len = self.trial_length(num_parameters)
    start_index = self.num_initial_tokens + trial_index * trial_len
    end_index = start_index + num_parameters
    return (start_index, end_index)

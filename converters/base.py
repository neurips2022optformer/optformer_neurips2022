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

"""Converts Study protos into various text formats."""
import abc
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np
from optformer_neurips2022.t5x import utils


from vizier import pyvizier as shared_pyvizier
from vizier.service import pyvizier as oss_pyvizier

Scalar = Union[int, float, str]
Value = Union[Scalar, Sequence[Scalar]]
StudyInfo = Dict[str, Value]
ParameterInfo = Dict[str, Value]
ParameterValueDict = Dict[str, Scalar]
ParameterValueList = List[Scalar]
ParameterValues = Union[ParameterValueDict, ParameterValueList]
ParameterIntValues = List[int]
MetricValueDict = Dict[str, Scalar]
MetricValueList = List[Scalar]
MetricValues = Union[MetricValueDict, MetricValueList]
AlgorithmInfo = Union[str, int]  # int corresponds to Algorithm ENUM

Aux = Dict[str, Any]  # Auxiliary information to be returned from converter.

# Token to denote the end of study information and the start of parameters'
# information in the config.
STUDY_PARAMETER_SEPARATION_TOKEN = '&'
# Token to denote separation between two parameters' information string in the
# study config.
PARAMETER_SEPARATION_TOKEN = '*'
# Token to denote end of old trial and start of new trial.
TRIAL_SEPARATION_TOKEN = '|'
# parameter_metric_separation_token: Token to denote separation between a
# trial's parameters and metrics.
PARAMETER_METRIC_SEPARATION_TOKEN = '*'

_MINIMUM_CONFIG_PROBABILITY = {
    'bbob': 1.0,
    'default': 0.1,
}


class ConvertedStudy(NamedTuple):
  inputs: str  # Conditioning inputs.
  target_inputs: str  # Inputs aligned with the targets.
  targets: str  # Targets.
  # The number of parameters with a non-fixed value in the search space. For a
  # conditional search space, we need to count all possible parameters.
  num_parameters: int
  aux: Aux
  num_permuted_trials: int = 0


DEFAULT_VIZIER_ALGORITHM = 0


def get_algorithm(sc,
                  default_algorithm = None):
  """Gets algorithm name/ID."""
  # Searches metadata for 'designer' key for generated studies.
  algo_info = sc.metadata.get('designer', default=None)
  if algo_info is None:
    # Returns standard algorithm ENUM for database studies.
    # WARNING: This could return 'DEFAULT' enum even for empty studies!
    return sc.algorithm
  else:
    algo_info = str(algo_info)
    if default_algorithm is not None and algo_info == default_algorithm:
      return DEFAULT_VIZIER_ALGORITHM
    else:
      return algo_info


class Converter(abc.ABC):
  """Converts Studies into text representations for language models.

  These strings will later be fed into a tokenizer, which is dependent on
  Transformer pipeline.
  """

  @abc.abstractmethod
  def study_to_texts(self,
                     study):
    raise NotImplementedError('Abstract method')

  def set_config(self, config):
    """Override configurations."""
    raise NotImplementedError('Abstract method')

  def pytrial_to_parameter_values(
      self, aux,
      trial):
    """Convert a trial to parameter values represented in integers."""
    raise NotImplementedError('Abstract method')

  def parameter_texts_to_trial(
      self, aux, parameter_texts):
    """Reverse the mapping from parameter value to text."""
    raise NotImplementedError('Abstract method')

  @property
  def trial_token_scheme(self):
    raise NotImplementedError('Abstract property')

  @property
  def num_quantized_values(self):
    """Number of quantized objective values."""
    raise NotImplementedError('Abstract property')

  def objective_value_to_int(
      self,
      value,
      aux,
      metric_name = None):
    """Convert a real objective value to a quantized integer."""
    raise NotImplementedError('Abstract method')

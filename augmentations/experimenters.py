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

"""Custom experimenters for OptFormer project."""
# pylint:disable=undefined-variable,invalid-name
import copy
import typing
from typing import Callable, List, NamedTuple, Optional, Tuple, Text, Sequence

import numpy as np
from optformer_neurips2022.augmentations import gp_bandit
import tensorflow as tf

from vizier.service import pyvizier


EXPORTED_PREDICTOR_DIR_FORMAT = ('/tmp/'
                                 'new_studies_for_eval/exported_gp_models/'
                                 'study_{study_guid}')


class StudyExtrapolaterSpec(NamedTuple):
  sstable_path: str
  study_index: int


def _create_study_extrapolater_experimenter(
    spec):
  """Create StudyExtrapolater experimenter from the spec."""
  # Load the study.
  table = sstable.SSTable(spec.sstable_path)
  for i, study_str in enumerate(table.itervalues()):
    if i == spec.study_index:
      break
  else:
    study_str = None
    raise ValueError(f'Requesting {spec.study_index}-th record in the '
                     f'SSTable {spec.sstable_path}, but it contains only '
                     f'{i} records.')
  study = study_str
  study.study_config.metadata.append(
      key_value_pb2.KeyValue(
          key='experimenter_type', value='StudyExtrapolater'))
  study_config = pyvizier.StudyConfig.from_proto(study.study_config)

  # Load the predictor.
  gp = gp_bandit.GaussianProcessModel(study_config)
  gp_export_dir = EXPORTED_PREDICTOR_DIR_FORMAT.format(
      study_guid=study.study_guid)
  predictor = load_gp_median_predictor(gp, gp_export_dir)
  experimenter = extrapolations.StudyExtrapolater(
      predictor=predictor, study_config=study_config)
  return experimenter


def _get_function_name(spec):
  """Extract function name from a spec."""
  if spec.HasField('bbob'):
    return config_pb2.BbobFunction.BbobFunctionName.Name(
        spec.bbob.function.name)
  elif spec.HasField('nasbench'):
    return config_pb2.NASBench.Dataset.Name(spec.nasbench.dataset)
  elif spec.HasField('combo'):
    return config_pb2.Combo.Dataset.Name(spec.combo.dataset)
  elif spec.HasField('sequin'):
    return config_pb2.Sequin.Dataset.Name(spec.sequin.dataset)
  elif spec.HasField('hpob'):
    return f'{spec.hpob.search_space_id}_{spec.hpob.dataset_id}'
  else:
    raise ValueError(f'Unsupported config: {spec}')


class TransformerExperimenter(object):
  """Specific Experimenter class used for the OptFormer project."""

  def __init__(self,
               spec,
               metadata_kv = (),
               always_maximize = True):
    if isinstance(spec, config_pb2.SingleObjectiveExperimenterSpec):
      self._experimenter = experimenter_factory.create_single_objective(spec)
    else:
      self._experimenter = _create_study_extrapolater_experimenter(spec)

    if (isinstance(spec, config_pb2.SingleObjectiveExperimenterSpec) and
        spec.HasField('hpob') and spec.hpob.seed):
      hpob_experimenter = typing.cast(hpob.HPOBExperimenter, self._experimenter)
      if spec.hpob.seed == 'random':
        hpob_seed = None
        rng = np.random
      else:
        hpob_seed = spec.hpob.seed
        rng = None
      self._init_trials = hpob_experimenter.get_initial_trials(
          hpob_seed=hpob_seed, rng=rng)
    else:
      self._init_trials = None

    has_minimize_goal = bbob_experimenters.has_single_minimize_goal(
        self._experimenter.study_config)
    if always_maximize and has_minimize_goal:
      self._experimenter = bbob_experimenters.SignFlippedExperimenter(
          self._experimenter)

    self._study_config = copy.deepcopy(self._experimenter.study_config)

    if isinstance(spec, config_pb2.SingleObjectiveExperimenterSpec):
      # Include function name in study_config.name.
      self._study_config.name = _get_function_name(spec)
      self._study_config.description = text_format.MessageToString(
          spec, as_utf8=True)
    for k, v in metadata_kv:
      self._study_config.metadata.append(key_value_pb2.KeyValue(key=k, value=v))

  def Evaluate(self, trial):
    return self._experimenter.Evaluate(trial)

  def XOpt(self):
    return self._experimenter.XOpt()

  @property
  def study_config(self):
    return copy.deepcopy(self._study_config)

  def DebugString(self):
    return self._experimenter.DebugString()

  @property
  def init_trials(self):
    """Get initial trials."""
    return self._init_trials


class GaussianProcessPredictor(tf.Module):
  """GP median predictor module that can be exported by tensorflow."""

  def __init__(self, model):
    self._model = model
    self._gp_layer = model.gp  # Assign the layer module to the object so
    # that all the variables are trackable during
    # exporting.
    self._keys = list(model.converter.features_shape.keys())
    input_signature = [
        tf.TensorSpec(shape=shape, dtype=model.dtype)
        for shape in model.converter.features_shape.values()
    ]
    self.predict_median_from_array = tf.function(
        self._predict_median_from_array, input_signature=input_signature)

  def _predict_median_from_array(self, *xs):
    """Exported TF function has to take a list of tensors."""
    xs_dict = {k: v for k, v in zip(self._keys, xs)}  # Make a dict from list.
    dist = self._model.predict_from_array(xs_dict)
    return dist.quantile(0.5)


def export_gp_median_predictor(model,
                               export_dir):
  """Export the GP predictor."""
  predictor = GaussianProcessPredictor(model)
  tf.saved_model.save(predictor, export_dir)


def load_gp_median_predictor(
    model,
    export_dir):
  """Load an exported GP model to predict the median."""
  predictor = tf.saved_model.load(export_dir)

  def median_predictor(trial):
    xs = model.converter.to_features([trial])
    med = predictor.predict_median_from_array(*list(xs.values()))
    return float(med)

  return median_predictor

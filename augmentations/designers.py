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

"""Designers used for data generation."""
# pylint:disable=wrong-arg-types,undefined-variable,invalid-name,missing-function-docstring

from typing import Sequence
from absl import logging
import pyglove as pg

from vizier.pyvizier import converters
from vizier.service import pyvizier

BASIC_DESIGNERS = [
    'designer_grid_search',
    'designer_grid_search_shuffled',
    'designer_random_search',
]

GP_DESIGNERS = [
    'designer_recursive_gp',
]

EVO_DESIGNERS = [
    'designer_reg_evo',
    'designer_hill_climb',
    'designer_hyper_lcs',
    'designer_eagle_strategy',
]

BASIC_AND_EVO_DESIGNERS = BASIC_DESIGNERS + EVO_DESIGNERS
MAIN_DESIGNERS = BASIC_AND_EVO_DESIGNERS + ['designer_recursive_gp']
ALL_DESIGNERS = BASIC_DESIGNERS + GP_DESIGNERS + EVO_DESIGNERS


class PyGloveDesigner(object):
  """PyGlove evolutionary algorithms wrapped as a Designer.

  Currently only supports flat search spaces, and ignores all
  scale_types. We also do not care about recovering from previous trials. If we
  do, please uncomment the bottom lines in Update().
  """

  def __init__(self, study_config,
               pyglove_algorithm):
    self._pyglove_algorithm = pyglove_algorithm
    self._dna_spec = pg.vizier.study_config_to_dna_spec(self._study_config)
    self._pyglove_algorithm.setup(self._dna_spec)
    self._pyconfig = study_config

    def create_metric_converter(mc):
      return converters.DefaultModelOutputConverter(
          mc,
          flip_sign_for_minimization_metrics=True,
          raise_errors_for_missing_metrics=False)

    # Convert objective metrics for now.
    self._converter = converters.DefaultTrialConverter([], [
        create_metric_converter(mc) for mc in self._pyconfig.objective_metrics
    ])

  def Suggest(self, num_suggestions):
    dna_list = [
        self._pyglove_algorithm.propose() for _ in range(num_suggestions)
    ]
    return [
        pg.vizier.dna_to_vizier_trial(dna, external_study=True)
        for dna in dna_list
    ]

  def Update(self, trials):
    pytrials = pyvizier.TrialConverter.from_protos(trials)
    labels = self._converter.to_labels_array(pytrials)
    for i in range(len(trials)):
      if not pytrials[i].is_completed or pytrials[i].infeasible:
        logging.warning('Skipping #%d trial: %s', i, trials[i])
        continue
      dna = pg.vizier.vizier_trial_to_dna(trials[i], self._dna_spec)
      # original_metadata = dict(trials[i].metadata)
      reward = tuple(labels[i])
      self._pyglove_algorithm.feedback(dna, reward)
      # if pg.ne(dna.metadata, original_metadata):
      #   trials[i] = pg.vizier.dna_to_vizier_trial(dna, external_study=True)

  def DebugString(self):
    return ('PyGlove Designer with algorithm: {} \n StudyConfig: {} \n '
            'DNASpec: {}').format(self._pyglove_algorithm, self._study_config,
                                  self._dna_spec)

  def Reset(self):
    self._pyglove_algorithm = self._pyglove_algorithm.clone(deep=True)


DesignerType = PyGloveDesigner


def create_pyglove_designer(
    name,
    study_config,
    mutator_factory=pg.evolution.mutators.Uniform,
):
  """PyGlove designer creation."""
  mutator = mutator_factory()
  if name == 'designer_reg_evo':
    pyglove_algorithm = pg.evolution.regularized_evolution(
        mutator=mutator, population_size=25, tournament_size=5)
  elif name == 'designer_hill_climb':
    pyglove_algorithm = pg.evolution.hill_climb(
        mutator=mutator, batch_size=1, init_population_size=1)

  return PyGloveDesigner(
      study_config=study_config, pyglove_algorithm=pyglove_algorithm)


def create_designer(name, study_config):
  if name in ['designer_reg_evo', 'designer_hill_climb']:
    return create_pyglove_designer(name, study_config)
  else:
    return create_designer(name, study_config)

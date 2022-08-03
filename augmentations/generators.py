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

"""Contains code for generating new studies."""
# pylint:disable=wrong-arg-types,undefined-variable
import random
from typing import Text, Callable, Sequence, Any
from absl import logging
import numpy as np

from optformer_neurips2022.augmentations import designers as designers_lib
from optformer_neurips2022.augmentations import experimenters
from optformer_neurips2022.augmentations import randomizers as randomizers_lib

from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
from vizier.pyvizier import converters as vizier_converters

from vizier.service import pyvizier


def generate_study_from_designer(
    spec,
    designer_name,
    num_trials,
    num_eval_trials = 0,
    always_maximize = True,
    use_init_trials = True):
  """Generates new studies with CLIFed Vizier Designers.

  Args:
    spec: Configuration spec to generate the experimenter.
    designer_name: Name of designer.
    num_trials: Length of optimization.
    num_eval_trials: Number of evaluation trials to append to the study. These
      are not used to update the designer. Can be used to analyze designer
      output distributions.
    always_maximize: always set goal to be maximizing. This optional is for
      backward-compatibility as old models are trained with maximization
      experimenters only.
    use_init_trials: use initial trials provided by the experimenter if
      available.

  Returns:
    A generated study.

  """
  experimenter = experimenters.TransformerExperimenter(
      spec=spec,
      metadata_kv=[('designer', designer_name)],
      always_maximize=always_maximize)

  init_trials = experimenter.init_trials if use_init_trials else None
  if init_trials:
    num_trials -= len(init_trials)

  if designer_name == 'designer_reg_evo':
    designer_name = 'py_designer_pyglove_regevo'
  elif designer_name == 'designer_hill_climb':
    designer_name = 'py_designer_pyglove_hill_climb'
  generator = create_suggestion_manager(designer_name,
                                        experimenter.study_config)
  runner = StudyRunner(
      generator,
      batch_num_trials=[1] * num_trials,
      num_trials_to_complete=num_trials,
      init_trials=init_trials)
  trials = list(runner.run_one_study(experimenter))
  for i in range(num_eval_trials):
    eval_trial = generator.suggest(1)[0]
    eval_trial.id = num_trials + 1 + i
    eval_trial.CopyFrom(evaluate_trial(experimenter, eval_trial))
    eval_trial.metadata.add(key='eval_id', value=str(i + 1))
    trials.append(eval_trial)

  return pyvizier.StudyWithTrials(
      problem=experimenter.study_config, trials=trials)


def study_generator_wrapper(
    spec_factory,
    study_generator
):
  """A wrapper of the study_generator to create a new spec in every call."""
  return study_generator(spec_factory())


tfd = tfp.distributions


@tf.function(jit_compile=True)
def sample_k(logits, k):
  """Sample k indices without replacement.

  Args:
    logits: Array of logits in float64.
    k: number of samples.

  Returns:
    Indices for the k samples without replacement, as int32 Tensor.
  """
  gumbel = tfd.Gumbel(
      tf.constant(0., dtype=tf.float64),
      tf.constant(1., dtype=tf.float64),
  )
  z = gumbel.sample(logits.shape)
  return tf.nn.top_k(logits + z, k).indices


def hpob_shuffled_study_generator(
    seed,
    max_num_trials = 300,
    strictness = 1e-2):
  """Generates a hpob study with trial orders shuffled.

  Trials are sampled without replacement from a multinomial distribution with
  logit=perturbed ranks. Good trials are likely to come before bad trials.


  Args:
    seed: Not a random seed. Each "seed" deterministically corresponds to a
      search_space, dataset_id pair.
    max_num_trials: Number of trials in the generated study.
    strictness: Higher value means less randomness. 1e-3 is almost random. 1e-1
      is almost fully sorted.

  Returns:
    Hpob study with trial orders shuffled.
  """
  try:
    container, search_space_id, dataset_id = randomizers_lib.hpob_randomizer(
        seed=seed)
    study = container.get_study(search_space_id, dataset_id)
    # converter returns labels as bigger = better
    converter = vizier_converters.TrialToArrayConverter.from_study_config(
        study.problem, flip_sign_for_minimization_metrics=True)
    # Flip the signs so that smaller = better
    labels = -converter.to_labels(study.trials).squeeze()

    # Convert ranks to logits. Better labels = smaller ranks = higher logits
    ranks = stats.rankdata(labels)
    logits = -ranks * strictness

    sampled_indices = sample_k(logits, min(labels.size, max_num_trials))
    trial_arr = np.array(study.trials)[sampled_indices]
    study.trials.clear()
    study.trials.extend(trial_arr)
  except:
    return pyvizier.StudyWithTrials()
  return study


def default_study_generator(
    seed,
    num_trials = 300,
    designers = tuple(designers_lib.MAIN_DESIGNERS),
    randomizers = (
        randomizers_lib.bbob_experimenter_config_randomizer,)
):
  """Generates a random Study from a random seed.

  This is the main function to modify when experimenting with different dataset
  generation.

  Args:
    seed: Seed randomizer.
    num_trials: Length of study.
    designers: Sequence of designer names to be used. Designers will occur
      equally often via cycling over the seeds, in order to ensure data is fair.
    randomizers: Sequence of functions mapping seed to experimenter configs.

  Returns:
    A Study proto generated from the seed. If failed, returns empty study.
  """
  try:
    rng = random.Random(seed)
    randomizer = rng.choice(randomizers)
    config = randomizer(seed=seed)
    logging.info('Config chosen: %s', config)
    designer_index = seed % len(designers)
    designer_name = designers[designer_index]
    logging.info('Designer chosen: %s', designer_name)
    study = generate_study_from_designer(
        spec=config, designer_name=designer_name, num_trials=num_trials)
  except:
    study = pyvizier.StudyWithTrials()
  return study

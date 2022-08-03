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

"""Configuration randomizers."""
# pylint:disable=undefined-variable,invalid-name
import random
from typing import Tuple, Optional, Callable, Sequence


ExperimenterConfigRandomizer = Callable[
    Ellipsis, config_pb2.SingleObjectiveExperimenterSpec]

ALL_BBOB_FUNCTIONS = tuple(config_pb2.BbobFunction.BbobFunctionName.values())
ALL_NOISE_TYPES = tuple(config_pb2.BbobNoise.BbobNoiseType.values())

MAX_INT32 = 2147483647


def bbob_experimenter_config_randomizer(
    bbob_functions = ALL_BBOB_FUNCTIONS,
    bbob_dimension_min = 1,
    bbob_dimension_max = 20,
    rotation_seed_min = 0,
    rotation_seed_max = MAX_INT32,
    sparsity_min = 0.0,
    sparsity_max = 0.0,
    add_shift = True,
    discretize = True,
    noise_types = ALL_NOISE_TYPES,
    seed = None,
    rng = None
):
  """Produces random specs for BBOB experimenter generation.

  Args:
    bbob_functions: Set of BBOB functions to sample from. Must not be empty.
    bbob_dimension_min: minimum number of BBOB function input dimensions.
    bbob_dimension_max: maximum number of BBOB function input dimensions
      (inclusive).
    rotation_seed_min: minimum seed for rotation matrix generation. NOTE: Adding
      this feature significantly slows down the benchmark. Unused by default.
    rotation_seed_max: maximum seed for rotation matrix generation (inclusive).
    sparsity_min: minimum sparsity ratio (ratio of number of true latent
      parameters to number of total parameters). When zero, this does nothing.
      Sparsity is induced by adding new "sparse" parameters and increasing
      dimensionality. Disabled for now to prevent dimensionality blowup.
    sparsity_max: maximum sparsity ratio.
    add_shift: Add shifting to the input (so that the optimum is no longer the
      0-vector).
    discretize: For each parameter, it may stay continuous, or become a DISCRETE
      or CATEGORICAL parameter, with a random number of feasible points.
    noise_types: Set of noise types to sample from. To remove noise, use the
      UNSET enum or send empty sequence.
    seed: RNG seed. This allows generation of the exact same config if given the
      same seed. NOTE: Experimenter output might still be different every study,
        as this seed does NOT control the randomness of evaluation.
    rng: RNG object. Only one of the seed and rng can be not None.

  Returns:
    Randomly generated BBOB experimenter spec.
  """
  if bbob_dimension_min > bbob_dimension_max:
    raise ValueError(f'bbob_dimension_min ({bbob_dimension_min}) must not be '
                     f'greater than bbob_dimension_max ({bbob_dimension_max}).')

  if seed is None:
    rng = rng or random
  elif rng is None:
    rng = random.Random(seed)
  else:
    raise ValueError('Only one of the seed and rng can be not None.')

  bbob = config_pb2.BbobExperimenterSpec()

  # Guard against Unimplemented/UNSET functions.
  possible_bbob_functions = list(bbob_functions)
  for deprecated in [config_pb2.BbobFunction.BbobFunctionName.UNSET, 8, 10, 15]:
    if deprecated in possible_bbob_functions:
      possible_bbob_functions.remove(deprecated)

  bbob.function.name = rng.choice(possible_bbob_functions)
  bbob.dimension = rng.randint(bbob_dimension_min, bbob_dimension_max)

  # bbob seed refers to rotation matrix generation.
  bbob.seed = rng.randint(rotation_seed_min, rotation_seed_max)
  bbob.sparsity = rng.uniform(sparsity_min, sparsity_max)

  if add_shift:
    shifts = [rng.uniform(-4.0, 4.0) for _ in range(bbob.dimension)]
    bbob.shift.fixed_shift.CopyFrom(
        config_pb2.BbobShift.FixedShift(shifts=shifts))

  if discretize:
    discretizations = bbob.discretization_map.discretizations
    for d in range(bbob.dimension):
      parameter_type = rng.choice([
          vizier_pb2.ParameterConfig.DOUBLE,
          vizier_pb2.ParameterConfig.DISCRETE,
          vizier_pb2.ParameterConfig.CATEGORICAL
      ])
      if parameter_type == vizier_pb2.ParameterConfig.DOUBLE:
        pass
      else:
        discretizations[d].num_feasible_points = rng.randint(2, 8)
        if parameter_type == vizier_pb2.ParameterConfig.DISCRETE:
          discretizations[d].map_to_string = False
        elif parameter_type == vizier_pb2.ParameterConfig.CATEGORICAL:
          discretizations[d].map_to_string = True

  if noise_types:
    possible_noise_types = list(noise_types)
    # Remove deprecated noise types.
    for deprecated in [
        config_pb2.BbobNoise.BbobNoiseType.ADDITIVE_GAUSSIAN,
        config_pb2.BbobNoise.BbobNoiseType.ADDITIVE_UNIFORM
    ]:
      if deprecated in possible_noise_types:
        possible_noise_types.remove(deprecated)
    bbob.noise.type = rng.choice(possible_noise_types)
  return config_pb2.SingleObjectiveExperimenterSpec(bbob=bbob)


ALL_COMBO_FUNCTIONS = tuple(config_pb2.Combo.Dataset.values())


def combo_experimenter_config_randomizer(
    combo_functions = ALL_COMBO_FUNCTIONS,
    combo_dimension_min = 1,
    combo_dimension_max = 20,
    seed = None,
    rng = None
):
  """Produces random specs for COMBO experimenter generation.

  Args:
    combo_functions: Set of COMBO functions to sample from. Must not be empty.
    combo_dimension_min: minimum number of COMBO function input dimensions.
    combo_dimension_max: maximum number of COMBO function input dimensions
      (inclusive).
    seed: RNG seed. This allows generation of the exact same config if given the
      same seed. NOTE: Experimenter output might still be different every study,
        as this seed does NOT control the randomness of evaluation.
    rng: RNG object. Only one of the seed and rng can be not None.

  Returns:
    Randomly generated COMBO experimenter spec.
  """
  if combo_dimension_min > combo_dimension_max:
    raise ValueError(
        f'combo_dimension_min ({combo_dimension_min}) must not be '
        f'greater than combo_dimension_max ({combo_dimension_max}).')

  if seed is None:
    rng = rng or random
  elif rng is None:
    rng = random.Random(seed)
  else:
    raise ValueError('Only one of the seed and rng can be not None.')

  combo = config_pb2.Combo()
  possible_combo_functions = list(combo_functions)
  if config_pb2.Combo.Dataset.UNSPECIFIED in possible_combo_functions:
    possible_combo_functions.remove(config_pb2.Combo.Dataset.UNSPECIFIED)
  combo.dataset = rng.choice(possible_combo_functions)
  combo.dimension = rng.randint(combo_dimension_min, combo_dimension_max)
  combo.task_id = rng.randint(0, 1000000000)
  return config_pb2.SingleObjectiveExperimenterSpec(combo=combo)


ALL_NASBENCH_FUNCTIONS = tuple(config_pb2.NASBench.Dataset.values())


def nasbench_experimenter_config_randomizer(
    nasbench_functions = (
        config_pb2.NASBench.Dataset.NASBENCH_201,),
    seed = None,
    rng = None
):
  """Produces random specs for NASBENCH experimenter generation.

  Args:
    nasbench_functions: Set of NASBENCH functions to sample from. Must not be
      empty.
    seed: RNG seed. This allows generation of the exact same config if given the
      same seed. NOTE: Experimenter output might still be different every study,
        as this seed does NOT control the randomness of evaluation.
    rng: RNG object. Only one of the seed and rng can be not None.

  Returns:
    Randomly generated NASBENCH experimenter spec.
  """

  if seed is None:
    rng = rng or random
  elif rng is None:
    rng = random.Random(seed)
  else:
    raise ValueError('Only one of the seed and rng can be not None.')

  nasbench = config_pb2.NASBench()
  possible_nasbench_functions = list(nasbench_functions)
  if config_pb2.NASBench.Dataset.UNSPECIFIED in possible_nasbench_functions:
    possible_nasbench_functions.remove(config_pb2.NASBench.Dataset.UNSPECIFIED)
  nasbench.dataset = rng.choice(possible_nasbench_functions)
  return config_pb2.SingleObjectiveExperimenterSpec(nasbench=nasbench)


ALL_SEQUIN_FUNCTIONS = tuple(config_pb2.Sequin.Dataset.values())


def sequin_experimenter_config_randomizer(
    sequin_functions = ALL_SEQUIN_FUNCTIONS,
    seed = None,
    rng = None
):
  """Produces random specs for COMBO experimenter generation.

  Args:
    sequin_functions: Set of COMBO functions to sample from. Must not be empty.
    seed: RNG seed. This allows generation of the exact same config if given the
      same seed. NOTE: Experimenter output might still be different every study,
        as this seed does NOT control the randomness of evaluation.
    rng: RNG object. Only one of the seed and rng can be not None.

  Returns:
    Randomly generated COMBO experimenter spec.
  """
  if seed is None:
    rng = rng or random
  elif rng is None:
    rng = random.Random(seed)
  else:
    raise ValueError('Only one of the seed and rng can be not None.')

  sequin = config_pb2.Sequin()
  possible_sequin_functions = list(sequin_functions)
  if config_pb2.Sequin.Dataset.UNSPECIFIED in possible_sequin_functions:
    possible_sequin_functions.remove(config_pb2.Sequin.Dataset.UNSPECIFIED)
  sequin.dataset = rng.choice(possible_sequin_functions)
  return config_pb2.SingleObjectiveExperimenterSpec(sequin=sequin)


def hpob_randomizer(mode = hpob.DEFAULT_TEST_MODE,
                    split = hpob.TEST,
                    seed = 0):
  """See hpob_experimtner_config_randomizer."""
  assert split in hpob.SPLITS
  handler = hpob_handler.HPOBHandler(
      root_dir=hpob.ROOT_DIR, mode=mode, surrogates_dir=hpob.SURROGATES_DIR)
  container = hpob.HPOBContainer(handler)
  ss_and_dataset_ids = list(container.dataset_keys(split))
  index = seed % len(ss_and_dataset_ids)
  search_space_id, dataset_id = ss_and_dataset_ids[index]
  return container, search_space_id, dataset_id


def hpob_experimenter_config_randomizer(
    mode = hpob.DEFAULT_TEST_MODE,
    split = hpob.TEST,
    na_policy = 'CONTINUOUS',
    categorical_policy = 'AS_CONTINUOUS',
    hpob_seed = 'random',
    seed = 0):
  """Produces specs for HPOB experimenter generation.

  Enumerates all dataset keys via grid search rather than randomly.

  Args:
    mode: HPOB mode. Should be `v3-train-augmented` when actually generating
      data. Currently defaults to test file.
    split: Either 'train', 'validation', 'test'.
    na_policy: See hpob.NaPolicy.
    categorical_policy: See hpob.CategoricalPolicy.
    hpob_seed: Must be one of ('', 'random', 'test0', 'test1', 'test2', 'test3',
      'test4'). Do not use seed trials if empty, randomly sample seed trials if
      is 'random', and otherwise uses the fixed seed trials from HPO-B.
    seed: RNG seed. This allows generation of the exact same config if given the
      same seed. NOTE: Experimenter output might still be different every study,
        as this seed does NOT control the randomness of evaluation.

  Returns:
    Generated HPOB experimenter spec.
  """
  _, search_space_id, dataset_id = hpob_randomizer(mode, split, seed)
  hpob_spec = config_pb2.HPOB(
      mode=mode, search_space_id=search_space_id, dataset_id=dataset_id,
      na_policy=na_policy, categorical_policy=categorical_policy,
      seed=hpob_seed)

  return config_pb2.SingleObjectiveExperimenterSpec(hpob=hpob_spec)

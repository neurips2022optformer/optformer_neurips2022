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

"""Utility class and methods to run inference from a trained model."""
# pylint:disable=undefined-variable
import copy
import glob
import os
import re
import tempfile
import typing
from typing import Any, Literal, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from absl import logging
from flax.core.frozen_dict import FrozenDict
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from optformer_neurips2022 import converters
from optformer_neurips2022.t5x import models as vizier_models
from optformer_neurips2022.tasks import t5_tasks

import seqio
from t5x import decoding
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
import tensorflow as tf

from vizier.service import pyvizier

Aux = converters.Aux
Batch = Mapping[str, jnp.ndarray]


INFERENCE_STUDY_CONVERTER_KWARGS = {
    # Do not filter studies.
    'algorithm_list': None,
    'study_filter': None,
    'min_trials': 0,
    'max_trials': 1000,
    'discard_const_objective': False,
    # Disable random data augmentation.
    'trial_permutation_probability': 0.,
    'min_objective_after_norm': 0.2,
    'max_objective_after_norm': 0.8,
    'rand_objective_scale_range': None,
    'minimum_config_per_study': False,
}


_DEFAULT_GIN_SEARCH_PATHS = ()


_DEFAULT_GIN_PATTERNS_TO_SKIP = [
    'train_script\\..*',
    # Deprecated configurations.
    'partitioning\\.PjitPartitioner\\.parameter_partitioning_dims.*',
    '.*\\.with_inputs =.*',
    '.*\\.WITH_INPUTS =.*',
]


def update_sequence(sequence,
                    index,
                    value):
  """Insert value to sequence at index."""
  index = jnp.asarray(index, dtype=jnp.int32)
  value = jnp.expand_dims(jnp.asarray(value, dtype=sequence.dtype), axis=1)
  one_hot = jax.nn.one_hot(index, sequence.shape[1], dtype=sequence.dtype)
  new_sequence = sequence * (1 - one_hot) + value * one_hot
  return new_sequence


def logits_to_log_probs(logits):
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_probs = logits - logits_sum
  return log_probs


def config_from_model_dir(model_dir,
                          gin_patterns_to_skip):
  """Modify the config file in a model dir and write it to a local copy."""
  config_file = os.path.join(model_dir, 'config.gin')

  with open(config_file, 'r') as f:
    config_str = f.read()

  # Remove ending "\\" in commented lines.
  modified = []
  for line in config_str.split('\n'):
    cur_line = line + ''  # Make a copy.
    if cur_line.strip().startswith('#'):
      while cur_line.strip().endswith('\\'):
        cur_line = cur_line.rstrip()[:-1]
    modified.append(cur_line)
  config_str = '\n'.join(modified)

  # Remove line continuation.
  config_str = config_str.replace('\\\n', ' ')

  modified = []
  for line in config_str.split('\n'):
    # Comment lines matching any of the given list of patterns.
    for pattern in gin_patterns_to_skip:
      if re.fullmatch(pattern, line):
        modified.append('# ' + line)
        break
    else:
      modified.append(line)
  config_str = '\n'.join(modified)

  local_config_file = tempfile.NamedTemporaryFile(
      mode='wt', prefix='config_', suffix='.gin', dir='/tmp', delete=False)
  local_config_file.write(config_str)
  local_config_file.close()

  return local_config_file.name


def _verify_indices_shapes(batch, indices):
  batch_size = batch['decoder_input_tokens'].shape[0]
  if batch_size != len(indices) or indices.ndim != 1:
    raise ValueError('Indices array must be 1-dimensional and match the '
                     'length of decoder_input_tokens.')


class InferenceModel(object):
  """Simple wrapper to load pretrained model and run prediction."""
  infer_task_count: int = 0

  def __init__(
      self,
      model,
      checkpoint_path,
      model_dir,
      params,
      train_state_axes,
      partitioner,
      batch_size,
      vocab,
      task_feature_lengths,
      num_initial_tokens = 1,  # Number of initial tokens in the target
                                    # string before first trial starts.
      vocab_index_from = 32000,  # First vocabulary index in prediction.
      dataset_builder = None):
    self._model = model
    self._checkpoint_path = checkpoint_path
    self._model_dir = model_dir
    self._params = params
    self._train_state_axes = train_state_axes
    self._partitioner = partitioner
    device_count = jax.device_count()
    batch_size = (batch_size + device_count - 1) // device_count * device_count
    self._batch_size = batch_size  # A multiple of device_count.
    self._vocab = vocab
    self._task_feature_lengths = task_feature_lengths
    self._num_initial_tokens = num_initial_tokens
    self._dataset_builder = (dataset_builder if dataset_builder
                             else self._make_dataset_builder())
    self._vocab_index_from = vocab_index_from
    self._vocab_index_to = (
        vocab_index_from + self.study_converter.num_quantized_values)
    self._rng = random.PRNGKey(0)

    self._partitioned_compute_logits_from_batch = self._partitioner.partition(
        self._compute_logits_from_batch,
        in_axis_resources=(train_state_axes.params,
                           partitioning.PartitionSpec('data',)),
        out_axis_resources=partitioning.PartitionSpec('data',),
        static_argnums=[2])

    self._partitioned_model_predict_fn = self._partitioner.partition(
        self._model_predict_fn,
        in_axis_resources=(self._train_state_axes.params,
                           partitioning.PartitionSpec('data',), None, None),
        out_axis_resources=partitioning.PartitionSpec('data',),
        static_argnums=[4, 5, 6])

  @classmethod
  def from_checkpoint(
      cls,
      checkpoint_path_or_model_dir,
      batch_size = 1,
      model_gin_file = None,
      gin_patterns_to_skip = None,
      gin_search_paths = _DEFAULT_GIN_SEARCH_PATHS,
      overwrite_gin_files = None,
      overwrite_gin_bindings = None,
  ):
    """Create an inference model from a checkpoint path or model directory.

    The model_gin_file, or if None, the config file from the model directory
    will be applied first except the training script related configs. Then if
    overwrite_gin_files or overwrite_gin_bindings are provided, they will be
    applied to overwrite the configurations.

    Args:
      checkpoint_path_or_model_dir: checkpoint path or model directory.
      batch_size: batch size.
      model_gin_file: model gin file if not None, otherwise use the default file
        in the model directory.
      gin_patterns_to_skip: sequence of gin string patterns with which lines in
        the model gin file will be skipped.
      gin_search_paths: paths that will be searched for gin files.
      overwrite_gin_files: paths to gin config files to be parsed. Files will be
        parsed in order with conflicting settings being overridden by later
        files. Paths may be relative to paths in `gin_search_paths`.
      overwrite_gin_bindings: individual gin bindings to be applied after the
        gin files are parsed. Will be applied in order with conflicting settings
        being overridden by later ones.

    Returns:
      An InferenceModel instance.
    """
    gin_patterns_to_skip = list(
        gin_patterns_to_skip) if gin_patterns_to_skip else []
    for pattern in _DEFAULT_GIN_PATTERNS_TO_SKIP:
      if pattern not in gin_patterns_to_skip:
        gin_patterns_to_skip.append(pattern)

    checkpoint_path_or_model_dir = os.path.normpath(
        checkpoint_path_or_model_dir)
    dirname = checkpoint_path_or_model_dir.split(os.sep)[-1]
    if not dirname.startswith('checkpoint_'):
      # The input is a model directory.
      model_dir = checkpoint_path_or_model_dir
      # Look for the latest checkpoint in the directory.
      checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint_*'))
      checkpoints = [path for path in checkpoints
                     if re.fullmatch('.*checkpoint_[\\d]+$', path)]
      checkpoint_path = sorted(checkpoints,
                               key=lambda s: int(s.split('_')[-1]))[-1]
      logging.info('Model dir: %s', model_dir)
      logging.info('Found the latest checkpoint path in the model dir: %s',
                   checkpoint_path)
    else:
      # The input is a model checkpoint path under the model directory.
      checkpoint_path = checkpoint_path_or_model_dir
      model_dir = os.path.dirname(checkpoint_path_or_model_dir)
      logging.info('Model dir: %s', model_dir)
      logging.info('Checkpoint path: %s', checkpoint_path)
    checkpoint_step = checkpoint_path.split('_')[-1]

    if model_gin_file is None:
      model_gin_file = config_from_model_dir(model_dir, gin_patterns_to_skip)

    gin_files = [model_gin_file]
    if overwrite_gin_files:
      gin_files.extend(overwrite_gin_files)
      logging.info('Model config will be overridden by the following files:\n'
                   '%s', overwrite_gin_files)
    gin_bindings = (list(overwrite_gin_bindings)
                    if overwrite_gin_bindings else [])
    gin_bindings.extend([
        'PjitPartitioner.num_partitions = 1',
        'MODEL_DIR = "/tmp/t5x"',
        'RETURN_ALL_DECODES = False',
        f'CHECKPOINT_PATH = "{checkpoint_path}"',
        f'STEP_OFFSET = {checkpoint_step}'
    ])
    logging.info('Model config will be overridden by the following bindings'
                 ':\n%s', overwrite_gin_bindings)
    with gin.unlock_config():
      gin_utils.parse_gin_flags(gin_search_paths=gin_search_paths,
                                gin_files=gin_files,
                                gin_bindings=gin_bindings)
    logging.info('Gin Configuration to restore model:\n%s', gin.config_str())

    vocabulary = gin.query_parameter('%VOCABULARY').scoped_configurable_fn()

    model = gin.query_parameter('%MODEL').scoped_configurable_fn()
    partitioner = gin.get_configurable(partitioning.PjitPartitioner)()
    checkpoint_config = gin.get_configurable(utils.RestoreCheckpointConfig)(
        path=checkpoint_path)
    task_feature_lengths = gin.query_parameter('%TASK_FEATURE_LENGTHS')
    for k, v in task_feature_lengths.items():
      # If the length is a gin ConfigurableReference, get its value.
      if hasattr(v, 'scoped_configurable_fn'):
        task_feature_lengths[k] = v.scoped_configurable_fn()
    if 'target_inputs' not in task_feature_lengths:
      # Backward compatibility with old model configs without target_inputs.
      task_feature_lengths['target_inputs'] = task_feature_lengths['targets']
    sequence_length = task_feature_lengths
    # Restore model checkpoint.
    logging.info('Restoring model parameters from %s', checkpoint_path)
    train_state, train_state_axes = restore_train_state(
        model, batch_size, sequence_length, partitioner, checkpoint_config,
        from_scratch=False)
    logging.info('Model parameters restored.')

    kwargs = dict(
        model=model,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        params=train_state.params,
        train_state_axes=train_state_axes,
        partitioner=partitioner,
        batch_size=batch_size,
        vocab=vocabulary,
        task_feature_lengths=task_feature_lengths
    )

    try:
      num_initial_tokens = gin.query_parameter('%NUM_INITIAL_TOKENS')
      kwargs['num_initial_tokens'] = num_initial_tokens
    except ValueError:
      num_initial_tokens = None
      logging.warning('NUM_INITIAL_TOKENS is not found in the model config '
                      'file. Using the default value.')
    return cls(**kwargs)

  @classmethod
  def increase_infer_task_count(cls):
    cnt = cls.infer_task_count
    cls.infer_task_count += 1
    return cnt

  @property
  def model(self):
    return self._model

  @property
  def vocab(self):
    return self._vocab

  @property
  def vocab_index_from(self):
    return self._vocab_index_from

  @property
  def study_converter(self):
    return self._dataset_builder.study_converter

  @property
  def study_aux_list(self):
    return self._dataset_builder.study_aux_list

  def _make_dataset_builder(self):
    """Create a dataset builder through gin configuration.

    The gin configuration must include the following configurations:
    t5_tasks.add_tasks, STUDY_CONVERTER.

    Returns:
      t5_tasks.DatasetFromStudy object and the associated study converter.
    """
    cnt = self.increase_infer_task_count()
    add_tasks_fn = gin.get_configurable(t5_tasks.add_tasks)
    study_converter = gin.query_parameter(
        '%STUDY_CONVERTER').scoped_configurable_fn(
            **INFERENCE_STUDY_CONVERTER_KWARGS)
    task_name = f'infer_task_{cnt}'
    dataset_builder = t5_tasks.DatasetFromStudy(
        add_tasks_fn=add_tasks_fn,
        study_converter=study_converter,
        feature_converter_cls=self._model.FEATURE_CONVERTER_CLS,
        task_feature_lengths=self._task_feature_lengths,
        batch_size=self._batch_size,
        task_name=task_name)
    return dataset_builder

  def get_dataset(self, study_list
                  ):
    return self._dataset_builder.dataset(study_list)

  def _pad_batch(self, batch):
    """Pad the feature batch size to a multiple of the jax device count."""
    base = jax.device_count()
    num_examples = next(iter(batch.values())).shape[0]
    num_padding = (num_examples + base - 1) // base * base - num_examples
    if num_padding > 0:
      batch = {k: jnp.concatenate(
          [v, jnp.tile(v[-1:], [num_padding] + [1] * (v.ndim-1))])
               for k, v in batch.items()}
    return batch, num_examples

  def _split_batch(self, batch):
    """Split a large batch to small ones with a size upto self._batch_size."""
    num_examples = next(iter(batch.values())).shape[0]
    if num_examples <= self._batch_size:
      return [batch]
    batches = []
    for i in range(0, num_examples, self._batch_size):
      batches.append({
          k: v[i:min(num_examples, i + self._batch_size)]
          for k, v in batch.items()
      })
    return batches

  def compute_logits_from_batch(
      self,
      batch,
      restrict_vocab_index = True):
    """Compute the logits given a batch of features."""
    padded_batch, num_examples = self._pad_batch(batch)
    logits_list = []
    for one_batch in self._split_batch(padded_batch):
      logits_list.append(
          self._partitioned_compute_logits_from_batch(self._params, one_batch,
                                                      restrict_vocab_index))
    if len(logits_list) == 1:
      logits = logits_list[0]
    else:
      logits = jnp.concatenate(logits_list, axis=0)
    if num_examples < logits.shape[0]:
      logits = logits[:num_examples]
    return logits

  def _compute_logits_from_batch(
      self,
      params,
      batch,
      restrict_vocab_index = True):
    """Wrapper of model._compute_logits."""
    logits = self.model._compute_logits(params, batch)
    if restrict_vocab_index:
      logits = logits[:, :, self._vocab_index_from: self._vocab_index_to]
    return logits

  def compute_logits(
      self,
      study_list,
      trial_list = None,
      restrict_vocab_index = True,
      ):
    """Compute logits, with an optionally restricted vocabulary.

    Args:
      study_list: list of studies.
      trial_list: optional list of trials to append to the input studies.
      restrict_vocab_index: return the logits in the valid token value range.

    Returns:
      Logits array of shape [S, T, V] where S is the number of studies,
        T is the maximum target sequence length and V is the number of discrete
        function values.
      List of feature sequences batches, converted from the list of (appended)
        studies.
      List of study converter auxiliary output dicts, one per study.
    """
    # If trial_list is provided, append the trial to the study trial list.
    if trial_list is not None:
      if len(study_list) != len(trial_list):
        raise ValueError('Length of study_list does not match trial_list.')
      study_list = [self.pad_study(study, len(study.trials) + 1, trial)
                    for study, trial in zip(study_list, trial_list)]

    dataset = self.get_dataset(study_list).as_numpy_iterator()
    logits = []
    batches = []
    for batch in dataset:
      logits.append(self.compute_logits_from_batch(
          batch, restrict_vocab_index=restrict_vocab_index))
      batches.append(batch)
    logits = jnp.concatenate(logits)  # [study number, sequence len, vocab size]
    return logits, batches, self.study_aux_list

  def compute_log_probs_from_batch(
      self,
      batch,
      restrict_vocab_index = True,
      ):
    """Compute log probability, with an optionally restricted vocabulary."""
    logits = self.compute_logits_from_batch(
        batch, restrict_vocab_index=restrict_vocab_index)
    return logits_to_log_probs(logits)

  def compute_log_probs(
      self,
      study_list,
      restrict_vocab_index = True,
      ):
    """Compute log probability, with an optionally restricted vocabulary."""
    logits, batches, aux_list = self.compute_logits(
        study_list, restrict_vocab_index=restrict_vocab_index)
    return logits_to_log_probs(logits), batches, aux_list

  def predict_fun_logits(
      self,
      study_list,
      trial_list,
      restrict_vocab_index = True):
    """Predict the function distribution of a trial given a study.

    Args:
      study_list: list of studies containing observations. Assuming all the
        trials are completed.
      trial_list: list of trials to predict the function value given parameters.
      restrict_vocab_index: return the logits in the valid token value range.

    Returns:
      Logits array of shape [S, V] where S is the number of studies, V is the
        number of discrete function values.
    """
    logits_seq, _, _ = self.compute_logits(
        study_list, trial_list, restrict_vocab_index)  # S, T, V.

    # Find the index of the function in the last trial.
    scheme = self.study_converter.trial_token_scheme
    f_indices = []
    for study, aux in zip(study_list, self.study_aux_list):
      f_indices.append(scheme.fun_index_in_trial(
          num_parameters=self.num_parameters(aux),
          trial_index=len(study.trials)))  # Index of the input trial.
    # For every study i, take the logits at index f_indices[i].
    f_logits = jnp.take_along_axis(
        logits_seq, jnp.array(f_indices)[:, None, None], axis=1)[:, 1]  # S, V.
    return f_logits

  def pad_study(self,
                study,
                min_trials,
                trial_to_append = None
                ):
    """Pad study trials upto min_trials with trial_to_append if provided."""
    # The objective value in the trial_to_append is ignored.
    num_trials_to_append = min_trials - len(study.trials)
    if num_trials_to_append > 0:
      # Append trials.
      if trial_to_append is None:
        trial_to_append = self._dummy_trial(study)
      else:
        trial_to_append = self._set_dummy_fun_value(study, trial_to_append)
      study = copy.deepcopy(study)
      study.trials.extend([trial_to_append] * num_trials_to_append)
    return study

  def _dummy_trial(self, study):
    """Make a trial with dummy parameter and function values."""
    py_trial = pyvizier.Trial()
    sc = pyvizier.StudyConfig.from_proto(study.study_config)
    for pc in sc.search_space.parameters:
      if pc.type in [
          pyvizier.ParameterType.CATEGORICAL, pyvizier.ParameterType.DISCRETE
      ]:
        value = pc.feasible_values[0]
      else:
        value = pc.bounds[0]  # Minimum value.
      py_trial.parameters[pc.name] = pyvizier.ParameterValue(value=value)
    trial = pyvizier.TrialConverter.to_proto(py_trial)
    trial = self._set_dummy_fun_value(study, trial)
    return trial

  def _set_dummy_fun_value(self, study, trial):
    """Set a dummy function value that does not affect normalization."""
    sc = pyvizier.StudyConfig.from_proto(study.study_config)
    if len(sc.metric_information) != 1:
      raise ValueError('Study contains zero or multiple metric information.')
    metric_name = sc.metric_information.item().name
    if study.trials:
      ref_trial = pyvizier.TrialConverter.from_proto(study.trials[0])
      value = ref_trial.final_measurement.metrics[metric_name].value
    else:
      value = 1.0
    py_trial = pyvizier.TrialConverter.from_proto(trial)
    py_trial.complete(pyvizier.Measurement({metric_name: value}))
    trial = pyvizier.TrialConverter.to_proto(py_trial)
    return trial

  def _model_predict_fn(
      self,
      params,
      batch,
      rng,
      decode_until = None,
      decoder_params = None,
      return_all_decodes = False,
      num_decodes = 1):
    """Wrapper of model.predict_batch_with_aux to draw samples."""
    if decoder_params is None:
      decoder_params = dict()
    decoder_params = dict(decoder_params, decode_until=decode_until)
    # Vizier models' predict_batch_with_aux method has a different signature.
    model = typing.cast(vizier_models.VizierEncoderDecoderModel, self._model)
    model.set_decoding_algorithm(decoder_params['decoding_algorithm'])
    # Decoder params contains arguments for the decoding function.
    # once the decoding function is set as above, we remove 'decoding_algorithm'
    # because it is an invalid parameter to be passed to the decoding function.
    # NB : decoding_algorithm string cannot be naively passed as a parameter
    # because this function may be jitted.
    del decoder_params['decoding_algorithm']
    return model.predict_batch_with_aux(
        params, batch,
        rng=rng,
        decoder_params=decoder_params,
        return_all_decodes=return_all_decodes,
        num_decodes=num_decodes,
        prompt_with_targets=True)

  def _get_logits_mask(self,
                       batch_size,
                       sequence_len,
                       input_start_indices,
                       input_end_indices):
    """Create mask for logits when generating parameters."""
    aux_batch = self.study_aux_list[-batch_size:]
    num_parameters = np.array([self.num_parameters(aux) for aux in aux_batch],
                              dtype=np.int32)
    max_num_parameters = np.max(num_parameters)

    # Since we are doing a lot of indexing, declaring as a numpy array first.
    logits_mask = np.full(
        (batch_size, sequence_len, self.vocab.vocab_size),
        0.0)
    if jnp.max(input_end_indices - input_start_indices) > max_num_parameters:
      raise NotImplementedError('Multi-trial masking currently not enabled.')
    for p_ind in range(max_num_parameters):
      for i in range(batch_size):  # Example index.
        aux = aux_batch[i]
        j = min(p_ind, num_parameters[i]-1)  # Parameter index.
        p_config = list(aux['parameter_name_to_configs'].values())[j]

        # Find the value range.
        if p_config.type in [
            pyvizier.ParameterType.CATEGORICAL,
            pyvizier.ParameterType.DISCRETE
        ]:
          num_values = len(p_config.feasible_values)
        else:
          num_values = self.study_converter.num_quantized_values

        start_index = self._vocab_index_from
        end_index = start_index + num_values

        # First fill everything with the negative infinity value.
        # -1 because we want to mask the logits before generating the param.
        param_start_idx = input_start_indices[i] + j - 1
        logits_mask[i, param_start_idx, :] = decoding.NEG_INF
        # Selectively set the appropriate values to 0.0
        logits_mask[i, param_start_idx, start_index:end_index] = 0.0

    # convert from numpy array to jax array
    return jnp.asarray(logits_mask)

  def _sampling_with_parameter_range(
      self,
      batch,
      input_start_indices,
      input_end_indices,
      decoder_params = None,
      decoding_algorithm = 'temperature_sample',
      return_all_decodes = False,
      num_decodes = 1,
      ):
    """Sample parameter values by model's predict_batch_with_aux method."""
    # Verify shapes.
    _verify_indices_shapes(batch, input_start_indices)
    _verify_indices_shapes(batch, input_end_indices)

    # Make a new dict to avoid modifying the original one.
    batch = {k: jnp.asarray(v) for k, v in batch.items()}

    # We apply masking to decoder_input_tokens so that trials before
    # input_start_indices will be used as prompts.
    #
    # For an example below ("I" is the initial tokens in the target string):
    #         decoder_target_tokens: I a b * c | d e * f |  ...
    #          decoder_input_tokens: 0 I a b * c | d e * f  ...
    #                input position: 0 1 2 3 4 5 6 7 8 9 10 ...
    # To sample the 2nd trial (trial index = 1), apply mask to
    # decoder_input_tokens after the first trial,
    #                          mask: 1 1 1 1 1 1 1 0 0 0 0  ...
    # The parameter index range of the trial in the input sequence: [7, 9)
    # We return the samples in the target sequence at position 6 and 7.
    batch_size, seq_length = batch['decoder_input_tokens'].shape
    mask = jnp.arange(seq_length) < input_start_indices[:, None]

    # Save the original decoder input tokens
    decoder_input_tokens = batch['decoder_input_tokens']
    batch['decoder_input_tokens'] = batch['decoder_input_tokens'] * mask

    # Update decoder params to account for fact that we are doing beam search
    decoder_params = {} if decoder_params is None else decoder_params

    decode_until = input_end_indices.item() - 1
    decoder_params['max_decode_len'] = seq_length - 1

    batch['logits_mask'] = self._get_logits_mask(
        batch_size, seq_length, input_start_indices, input_end_indices)

    decoder_params['decoding_algorithm'] = decoding_algorithm
    rng, self._rng = random.split(self._rng)
    if decoder_params is not None:
      decoder_params = FrozenDict(decoder_params)
    padded_batch, num_examples = self._pad_batch(batch)

    full_samples, _ = self._partitioned_model_predict_fn(
        self._params, padded_batch, rng, decode_until, decoder_params,
        return_all_decodes, num_decodes)

    if num_examples < full_samples.shape[0]:
      full_samples = full_samples[:num_examples]

    # Reset the batch to the correct state
    # Remove the logits mask
    del batch['logits_mask']

    # Reset the original decoder input tokens.
    batch['decoder_input_tokens'] = decoder_input_tokens

    if return_all_decodes:
      batch = {k: jnp.repeat(v, num_decodes, axis=0)
               for k, v in batch.items()}

    # Sampled (target) sequence is shifted to the left by 1.
    sample_start_indices = input_start_indices - 1
    sample_end_indices = input_end_indices - 1

    # We update 'decoder_input_tokens' in this block
    samples = []
    for i, (start_index, end_index) in enumerate(
        zip(sample_start_indices, sample_end_indices)):
      if return_all_decodes:
        this_sample = full_samples[i, :, start_index:end_index]
        batch['decoder_input_tokens'] = batch['decoder_input_tokens'].at[:, (
            start_index + 1):(end_index + 1)].set(this_sample)
        samples.append(this_sample)
      else:
        this_sample = full_samples[i, start_index:end_index]
        batch['decoder_input_tokens'] = batch['decoder_input_tokens'].at[
            i, (start_index + 1):(end_index + 1)].set(this_sample)
        samples.append(this_sample)

    return samples, batch

  def _naive_multistep_sampling_with_parameter_range(
      self,
      batch,
      input_start_indices,
      num_trials = 1,
      decoder_params = None,
      return_all_decodes = False,
      num_decodes = 1,
      ):
    """Sample parameter values by calling compute_logits once per parameter."""
    # This function only sample valid parameter values.

    # Verify shapes.
    _verify_indices_shapes(batch, input_start_indices)
    batch_size, seq_length = batch['decoder_input_tokens'].shape

    decoder_params = decoder_params or {}
    param_temperature = jnp.asarray(decoder_params.get('temperature', 1.0))
    function_temperature = jnp.asarray(
        decoder_params.get('function_temperature', param_temperature))

    if not return_all_decodes:
      # This function draw independent samples. If not returning all decodes,
      # just sample one and return it.
      num_decodes = 1

    # Make a new dict to avoid modifying the original one.
    if num_decodes == 1:
      batch = {k: jnp.asarray(v) for k, v in batch.items()}
    else:
      # Repeat every study feature sequence `num_decodes` times.
      batch = {k: jnp.repeat(v, num_decodes, axis=0) for k, v in batch.items()}

    aux_batch = self.study_aux_list[-batch_size:]
    num_parameters = np.array([self.num_parameters(aux) for aux in aux_batch],
                              dtype=np.int32)
    scheme = self.study_converter.trial_token_scheme
    trial_lens = scheme.trial_length(num_parameters)
    max_num_parameters = np.max(num_parameters)

    input_param_start_indices = input_start_indices.copy()
    for trial_idx in range(num_trials):
      # Sample one parameter dimension at a time.
      for p_ind in range(max_num_parameters):
        logits = self.compute_logits_from_batch(batch,
                                                restrict_vocab_index=True)
        logits = logits / param_temperature

        input_index_list = []
        next_token_list = []
        for i in range(batch_size):  # Example index.
          aux = aux_batch[i]
          j = min(p_ind, num_parameters[i]-1)  # Parameter index.
          p_config = list(aux['parameter_name_to_configs'].values())[j]
          # Find the value range.
          if p_config.type in [
              pyvizier.ParameterType.CATEGORICAL,
              pyvizier.ParameterType.DISCRETE
          ]:
            num_values = len(p_config.feasible_values)
          else:
            num_values = self.study_converter.num_quantized_values

          input_index = input_param_start_indices[i] + j
          target_index = input_index - 1
          logits_subset = logits[
              i*num_decodes:(i+1)*num_decodes,
              target_index,
              :num_values]  # [num_decodes, num_values]

          # Sample the token.
          rng, self._rng = random.split(self._rng)
          next_token = random.categorical(rng, logits_subset).astype(jnp.int32)
          next_token += self._vocab_index_from  # [num_decodes]

          input_index_list.append(
              jnp.ones(num_decodes, dtype=jnp.int32) * input_index)
          next_token_list.append(next_token)

        # Insert the next token to the decoder_input_tokens at input index.
        batch['decoder_input_tokens'] = update_sequence(
            batch['decoder_input_tokens'],
            jnp.concatenate(input_index_list),
            jnp.concatenate(next_token_list))

      if trial_idx < num_trials - 1:
        # Sample the function value.
        logits = self.compute_logits_from_batch(batch,
                                                restrict_vocab_index=True)
        logits = logits / function_temperature

        input_index_list = []
        next_token_list = []
        for i in range(batch_size):  # Example index.
          num_values = self.study_converter.num_quantized_values
          input_index = input_param_start_indices[i] + num_parameters[i] + 1
          target_index = input_index - 1
          logits_subset = logits[
              i*num_decodes:(i+1)*num_decodes,
              target_index,
              :num_values]  # [num_decodes, num_values]

          # Sample the token.
          rng, self._rng = random.split(self._rng)
          next_token = random.categorical(rng, logits_subset).astype(jnp.int32)
          next_token += self._vocab_index_from  # [num_decodes]

          input_index_list.append(
              jnp.ones(num_decodes, dtype=jnp.int32) * input_index)
          next_token_list.append(next_token)

        # Insert the next token to the decoder_input_tokens at input index.
        batch['decoder_input_tokens'] = update_sequence(
            batch['decoder_input_tokens'],
            jnp.concatenate(input_index_list),
            jnp.concatenate(next_token_list))

      # Update indices.
      input_param_start_indices += trial_lens

    # Extract samples.
    decoder_input_tokens = batch['decoder_input_tokens'].reshape([
        batch_size, num_decodes, seq_length])
    input_end_indices = (input_start_indices +
                         num_parameters + trial_lens * (num_trials - 1))
    samples = []
    for i, (start_index, end_index) in enumerate(
        zip(input_start_indices, input_end_indices)):
      # [num_decodes, trial_length].
      sample = decoder_input_tokens[i, :, start_index:end_index]
      if return_all_decodes:
        samples.append(sample)
      else:
        samples.append(sample[0])

    return samples, batch

  def sample_parameters_of_next_trials(
      self,
      study_list,
      num_trials = 1,
      num_samples = 1,
      decoder_params = None,
      naive_sampling = True,
      decoding_algorithm = 'temperature_sample'
  ):
    """Sample parameters of the next trial given a study."""
    trial_indices = [len(study.trials) for study in study_list]
    return self.sample_parameters_in_trials(
        study_list=study_list,
        trial_indices=trial_indices,
        num_trials=num_trials,
        decoder_params=decoder_params,
        return_all_decodes=True,
        num_decodes=num_samples,
        naive_sampling=naive_sampling,
        decoding_algorithm=decoding_algorithm,
    )

  def sample_parameters_in_trials(
      self,
      study_list,
      trial_indices,
      num_trials = 1,
      decoder_params = None,
      return_all_decodes = False,
      num_decodes = 1,
      naive_sampling = True,
      decoding_algorithm = 'temperature_sample'
  ):
    """Sample parameter prediction at a given trial index.

    For an example below ("I" is the initial tokens in the target string):
            decoder_target_tokens: I a b * c | d e * f |  ...
             decoder_input_tokens: 0 I a b * c | d e * f  ...
                   input position: 0 1 2 3 4 5 6 7 8 9 10 ...
    To sample the 2nd trial (trial index = 1),
    the parameter index range in the input sequence: [7, 9).
    We return the samples in the target sequence at position 6 and 7.

    Args:
      study_list: study list.
      trial_indices: list of 0-based trial indices to sample the parameters at.
      num_trials: number of trials to sample.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return all samples or just the top-1.
      num_decodes: the number of beams to use in beam search.
      naive_sampling: naive sampling without using for loop. It restricts the
        sampled tokens to the valid parameter value range.
      decoding_algorithm: whether to decode with beam search or temperature
        sampling when not doing naive sampling.

    Returns:
      List of parameter samples of length num_studies, each with shape of
        (num_decodes, num_trials, num_params) if return_all_decodes is True,
        otherwise (num_trials, num_params)
      List of function samples of length num_studies if num_trials > 1,
        otherwise list of None. If num_trials > 1, each has a shape of
        (num_decodes, num_trials - 1) if return_all_decodes is True,
        otherwise (num_trials - 1,)
      List of feature sequences batches, converted from the list of (padded)
        studies.
      List of study converter auxiliary outputs, one per study.
    """
    if not isinstance(self._model, vizier_models.VizierEncoderDecoderModel):
      raise NotImplementedError('Indexing parameters only supports '
                                'VizierEncoderDecoderModel.')

    # Make sure every study has at least `trial` trials with final measurements.
    if len(study_list) != len(trial_indices):
      raise ValueError('Length of study_list must match trial_indices.')
    study_list = copy.deepcopy(study_list)
    for i, (study, trial_index) in enumerate(zip(study_list, trial_indices)):
      if len(study.trials) < trial_index:
        raise ValueError(f'Cannot sample {trial_index}-th when the study has '
                         f'only {len(study.trials)} trials.')
      study_list[i] = self.pad_study(study, min_trials=trial_index+num_trials)

    # Feed the study list to the dataset and process by batches.
    dataset = self.get_dataset(study_list).as_numpy_iterator()
    scheme = self.study_converter.trial_token_scheme
    study_idx = 0
    samples = []
    batches = []
    num_params_list = []
    for batch in dataset:
      batch_size = batch['decoder_input_tokens'].shape[0]
      aux_batch = self.study_aux_list[-batch_size:]

      # Find the starting and ending index in the decoder input in a study.
      input_start_indices = np.zeros(batch_size, dtype=np.int32)
      input_end_indices = np.zeros(batch_size, dtype=np.int32)
      for i in range(batch_size):
        num_params = self.num_parameters(aux_batch[i])
        num_params_list.append(num_params)
        trial_index = trial_indices[study_idx]
        study_idx += 1
        max_trials = self.max_trials(num_params)
        if trial_index >= max_trials:
          raise ValueError(f'The model supports sampling upto {max_trials-1}-th'
                           f' trial with {num_params} parameters, but a trial '
                           f'index of {trial_index} is given.')

        index_range = scheme.param_index_range_in_trial(num_params, trial_index)
        # Input sequence is shifted to the right by 1 token.
        input_start_indices[i] = index_range[0] + 1
        input_end_indices[i] = (
            index_range[1] + 1 +
            scheme.trial_length(num_params) * (num_trials - 1))

      # Sampling the batch.
      if naive_sampling:
        sample, batch = self._naive_multistep_sampling_with_parameter_range(
            batch, input_start_indices, num_trials=num_trials,
            decoder_params=decoder_params,
            return_all_decodes=return_all_decodes,
            num_decodes=num_decodes)
      else:
        sample, batch = self._sampling_with_parameter_range(
            batch, input_start_indices, input_end_indices,
            decoder_params=decoder_params,
            decoding_algorithm=decoding_algorithm,
            return_all_decodes=return_all_decodes,
            num_decodes=num_decodes)
      samples.extend(sample)
      batches.append(batch)

    param_samples, fun_samples = self._split_parameter_and_function_samples(
        samples, num_params_list, num_trials)
    return param_samples, fun_samples, batches, self.study_aux_list

  def _split_parameter_and_function_samples(
      self,
      samples_list,
      num_params_list,
      num_trials):
    """Split trial samples into parameter and function samples."""
    if len(samples_list) != len(num_params_list):
      raise ValueError('The length of samples_list must match the length of '
                       'num_params_list')
    scheme = self.study_converter.trial_token_scheme
    if num_trials == 1:
      # Samples are the single trial of parameters. No function samples.
      param_samples_list = [
          jnp.expand_dims(samples, -2) for samples in samples_list
      ]
      fun_samples_list = [None] * len(samples_list)
    else:
      param_samples_list = []
      fun_samples_list = []
      for samples, num_params in zip(samples_list, num_params_list):
        # Calculate the index of parameter and function offset.
        f_idx_trial_0 = scheme.fun_index_in_trial(num_params, 0)
        p_range_trial_0 = scheme.param_index_range_in_trial(num_params, 0)
        f_offset = f_idx_trial_0 - p_range_trial_0[0]
        trial_len = scheme.trial_length(num_params)

        # Let L be the trial length and P be the number of parameters.
        # [[0, 1,   ..., P-1],
        #  [L, L+1, ..., L+P-1],
        #  ...]
        p_indices = (jnp.arange(num_params)[None, :] +
                     jnp.arange(num_trials)[:, None] * trial_len)
        # Shape: ([num_decodes,] num_trials, num_params)
        param_samples_list.append(samples[Ellipsis, p_indices])

        # [f_offset, L+f_offset, ..., ]
        f_indices = jnp.arange(num_trials-1) * trial_len + f_offset
        # Shape: ([num_decodes,] num_trials-1)
        fun_samples_list.append(samples[Ellipsis, f_indices])

      if samples_list[0].ndim == 1:
        assert all([p.shape == (num_trials, num_p)
                    for p, num_p in zip(param_samples_list, num_params_list)])
        assert all([f.shape == (num_trials-1,)
                    for f in fun_samples_list])
      else:
        assert samples_list[0].ndim == 2
        num_decodes = samples_list[0].shape[0]
        assert all([p.shape == (num_decodes, num_trials, num_p)
                    for p, num_p in zip(param_samples_list, num_params_list)])
        assert all([f.shape == (num_decodes, num_trials-1)
                    for f in fun_samples_list])
    return param_samples_list, fun_samples_list

  def num_parameters(self, aux):
    """Number of the nonfixed parameters.

    This is also the number of parameter tokens of a trial in the target
    sequence. It could be different from the number of parameters in a study
    when study_converter._filter_fixed_parameters is True and there exist
    parameters with fixed values.

    Args:
      aux: study converter auxiliary output dict.

    Returns:
      Number of the non-fixed parameters.
    """
    return len(aux['parameter_name_to_configs'])

  def max_trials(self, num_parameters = None,
                 aux = None):
    """Compute the supported maximum number of trials to infer for a study."""
    if num_parameters is None:
      num_parameters = self.num_parameters(aux)
    scheme = self.study_converter.trial_token_scheme
    target_len = self._task_feature_lengths['targets']
    return scheme.max_trials(num_parameters, target_len)

  def fun_index_in_trial(self, trial_index,
                         num_parameters = None,
                         aux = None):
    """Index of the function token in a trial."""
    scheme = self.study_converter.trial_token_scheme
    if num_parameters is None:
      num_parameters = self.num_parameters(aux)
    return scheme.fun_index_in_trial(num_parameters, trial_index)

  def set_seed(self, seed):
    self._rng = random.PRNGKey(seed)

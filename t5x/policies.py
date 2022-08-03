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

"""Wrap a pretrained model into a policy."""
# pylint:disable=missing-function-docstring,unused-argument,undefined-variable,invalid-name
import collections
import copy
import functools
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple

from jax import random
import jax.numpy as jnp
import numpy as np

from optformer_neurips2022 import converters
from optformer_neurips2022.t5x import inference_utils

from vizier.service import pyvizier

# Default arguments to create an inference_model for the Transformer designer.
DEFAULT_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': (''),
    'batch_size': 32,  # Maximum batch size to fit in a GPU worker.
}
DEFAULT_DESIGNER_NAME = 'designer_vizier'
DEFAULT_RANKING_CONFIG = {'type': 'ei'}


def _maybe_to_py(x):
  return None if x is None else x.to_py()


class TransformerDesigner(object):
  """Wraps a trained Transformer model into a designer."""

  def __init__(
      self,
      study_config,
      inference_model,
      designer_name = DEFAULT_DESIGNER_NAME,
      temperature = 1.0,
      suggest_with_rerank = False,
      num_samples = 128,
      naive_sampling = True,
      decoding_algorithm = 'temperature_sample',
      ranking_config = None,
      study_converter_config = None,
      additional_policy_study_converter_config = False,
      policy_study_converter_config = None,
  ):
    """Create a Transformer policy.

    Args:
      study_config: experimenter study config.
      inference_model: transformer inference model.
      designer_name: name of algorithm name to imitate. If None or empty, use
        the designer_name field in study_config.
      temperature: policy output temperature.
      suggest_with_rerank: augmented policy, i.e., ranking multiple parameter
        suggestion samples with acquisition function if True.
      num_samples: number of parameter suggestions to sample before ranking.
      naive_sampling: whether to do naive sampling or use block decoding.
      decoding_algorithm: whether to decode with beam search or temperature
        sampling when not doing naive sampling.
      ranking_config: ranking config dict.
      study_converter_config: override study converter if provided.
      additional_policy_study_converter_config: apply additional config to
        override the study converter used for the prior policy to sample
        parameter suggestions.
      policy_study_converter_config: additional config dict for prior policy.
        Effective if additional_policy_study_converter_config is True.
    """
    self._study_config = copy.deepcopy(study_config)
    has_minimize_goal = (self._study_config.goal
                         == pyvizier.ObjectiveMetricGoal.MINIMIZE) or (
                             self._study_config.metric_information[0].goal
                             == pyvizier.ObjectiveMetricGoal.MINIMIZE)

    self._metric_flipped = False
    if has_minimize_goal:
      flip_metric_goals(self._study_config)
      self._metric_flipped = True

    if designer_name:
      while self._study_config.metadata:
        self._study_config.metadata.pop()
      self._study_config.metadata['designer'] = designer_name
    self._decoder_params = {'temperature': temperature}
    self._p_study_config = pyvizier.StudyConfig.from_proto(self._study_config)
    self._inference_model = inference_model
    self._vocab = inference_model.vocab
    self._suggest_with_rerank = suggest_with_rerank
    self._num_samples = num_samples
    self._ranking_config = ranking_config or DEFAULT_RANKING_CONFIG
    self._num_trials = 1
    self._naive_sampling = naive_sampling
    self._decoding_algorithm = decoding_algorithm

    if (self._suggest_with_rerank and
        self._ranking_config['type'] == 'thompson_sampling'):
      self._num_trials = self._ranking_config.get('num_trials', 1)
      self._decoder_params['function_temperature'] = self._ranking_config.get(
          'function_temperature', temperature)
    self._function_encoder_input_tokens = None

    self._study_converter = self._inference_model.study_converter
    config = dict(inference_utils.INFERENCE_STUDY_CONVERTER_KWARGS)
    if study_converter_config:
      config.update(study_converter_config)
    self._study_converter.set_config(config)

    if additional_policy_study_converter_config:
      # If there are additional configs for policy sampling, make a copy of the
      # encoder_input_tokens for function prediction.
      #
      # Make sure input string is deterministic.
      self._study_converter.set_config({'randomize_parameter_order': False})
      # self._study_converter._randomize_parameter_order = False
      # Make a copy of encoder input tokens when predict functions.
      self._function_encoder_input_tokens = self._make_function_encoder_input()

      self._study_converter.set_config(policy_study_converter_config)

    self._historical_study: pyvizier.StudyWithTrials = pyvizier.StudyWithTrials(
        study_config=self._study_config)
    self._suggest_history = collections.defaultdict(list)

  def _make_function_encoder_input(self):
    study = pyvizier.StudyWithTrials(problem=self._study_config)
    _, _, batch_list, _ = (
        self._inference_model.sample_parameters_of_next_trials(
            [study],
            num_samples=self._num_samples,
            naive_sampling=self._naive_sampling,
            num_trials=1,
            decoder_params=self._decoder_params,
            decoding_algorithm=self._decoding_algorithm))
    batch = batch_list[0]
    return batch['encoder_input_tokens']

  def Suggest(self, num_suggestions):
    if not self._suggest_with_rerank:
      return self.suggest_samples(num_suggestions)
    else:
      return self.suggest_with_rerank(num_suggestions)

  def _sample(
      self, num_samples
  ):
    """Return the samples of shape [num_samples, num_parameters] and batch.

    Args:
      num_samples: number of samples to return.

    Returns:
      A (num_samples, num_trials, num_params) array of parameter samples.
      A (num_samples, num_trials-1) array of function samples or None.
      The corresponding dict of feature sequences for the transformer model.
      Study converter auxiliary output dict.
    """
    study = pyvizier.StudyWithTrials(
        problem=self._historical_study.study_config,
        trials=self._historical_study.trials)

    # TODO: determine a proper function scaling / transformation
    # through study converter or by appending auxiliary trials.
    param_samples_list, fun_samples_list, batch_list, aux_list = (
        self._inference_model.sample_parameters_of_next_trials(
            study_list=[study],
            num_samples=num_samples,
            naive_sampling=self._naive_sampling,
            num_trials=self._num_trials,
            decoder_params=self._decoder_params,
            decoding_algorithm=self._decoding_algorithm))
    aux = aux_list[0]
    trial_ids = aux.get('trial_ids') or aux['trial_permutation']['trial_ids']
    if len(trial_ids) - self._num_trials != len(study.trials):
      raise ValueError('The number of converted trials - num_sampling_trials '
                       f'({len(trial_ids)} - {self._num_trials}) '
                       f'does not match the input study ({len(study.trials)}). '
                       'Some trials are not valid.')
    # Only one study in the output list.
    return (param_samples_list[0].to_py(), _maybe_to_py(fun_samples_list[0]),
            batch_list[0], aux_list[0])

  def suggest_samples(self, num_suggestions):
    if num_suggestions != 1:
      raise NotImplementedError('Suggest function only supports '
                                'num_suggestions = 1.')
    samples, _, _, aux = self._sample(1)  # [1, 1, num_parameters]
    return [self.token_sample_to_trial(samples[0, 0], aux)]

  def function_logits(self, batch, trial_idx,
                      aux):
    """Use feature batch to predict the function value of the samples."""

    f_batch = dict(batch)
    if self._function_encoder_input_tokens is not None:
      # Replace the encoder_input_tokens.
      expected_shape = self._function_encoder_input_tokens.shape
      if batch['encoder_input_tokens'].shape != expected_shape:
        raise ValueError(f"{batch['encoder_input_tokens'].shape} != "
                         f'{self._function_encoder_input_tokens.shape}')
      f_batch['encoder_input_tokens'] = self._function_encoder_input_tokens

    seq_logits = self._inference_model.compute_logits_from_batch(
        f_batch, restrict_vocab_index=True)  # [S, T, V].

    f_token_idx = self._inference_model.fun_index_in_trial(trial_idx, aux=aux)
    logits = seq_logits[:, f_token_idx]  # [S, V]
    return logits

  def token_sample_to_trial(self, samples,
                            aux):
    """Decodes samples into raw parameter values and make a trial."""
    if samples.ndim != 1:
      raise ValueError('Token samples should be a 1D array.')

    parameter_texts = [
        self._vocab.decode_tf([token]).numpy().decode() for token in samples
    ]

    p_trial = self._study_converter.parameter_texts_to_trial(
        aux=aux, parameter_texts=parameter_texts)
    return pyvizier.TrialConverter.to_proto(p_trial)

  def suggest_with_rerank(self,
                          num_suggestions):
    # Notation of the variable dimensions:
    #   S: number of samples.
    #   T: number of trials.
    #   P: number of parameters.
    #   V: number of the function value discretization levels.
    if num_suggestions != 1:
      raise NotImplementedError('Suggest function only supports '
                                'num_suggestions = 1.')
    trial_idx = len(self._historical_study.trials)

    # param_samples: [S, T, P]
    # fun_params: [S, T-1] if T > 1 else None
    param_samples, fun_samples, sample_feat_batch, aux = (
        self._sample(self._num_samples))
    # Convert function token index [0, V-1] to quantized integer in [0, Q-1].
    if fun_samples is not None:
      fun_samples = fun_samples - self._inference_model.vocab_index_from

    # Predict the function value of the samples.
    # [S, V].
    logits = self.function_logits(sample_feat_batch, trial_idx, aux)

    if 'type' not in self._ranking_config:
      raise ValueError('ranking_config must include a key of "type" when '
                       'suggest_with_rerank = True.')
    ranking_type = self._ranking_config['type']
    if ranking_type == 'expected_mean':
      # Select the sample with the highest mean.
      scores = self._expected_mean(logits)
    elif ranking_type == 'ucb':
      scores = self._ucb(logits)
    elif ranking_type == 'thompson_sampling':
      scores = self._thompson_sampling(fun_samples, logits, aux)
    elif ranking_type == 'ei':
      scores = self._ei(logits, sample_feat_batch, aux)
    elif ranking_type == 'pi':
      scores = self._pi(logits, sample_feat_batch, aux)
    else:
      raise ValueError('Unknown ranking type: "{ranking_type}".')

    selected_sample_idx = np.argmax(scores)
    selected_sample = param_samples[selected_sample_idx, 0]

    # Convert a vector of parameter values to a Vizier trial.
    trial = self.token_sample_to_trial(selected_sample, aux)

    # Bookkeeping for the debugging purpose.
    trial_samples = [
        self.token_sample_to_trial(s[0], aux) for s in param_samples
    ]
    self._suggest_history['trial_samples'].append(trial_samples)
    self._suggest_history['logits'].append(logits)
    self._suggest_history['scores'].append(scores)
    self._suggest_history['selected_sample_idx'].append(selected_sample_idx)
    self._suggest_history['selected_trial'].append(trial)

    return [trial]

  def _ucb(
      self,
      logits  # [S, V].
  ):
    """Compute the p-quantile of discrete function distributions."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    p = self._ranking_config.get('upper_bound')
    if p is None or not 0 <= p <= 1:
      raise ValueError(
          'ranking_config must include a key of "upper_bound" in [0, 1] when '
          'ranking_config["type"] = "ucb".')
    log_probs = inference_utils.logits_to_log_probs(logits)
    cdf = jnp.cumsum(jnp.exp(log_probs), 1)
    quantile = (cdf < p).sum(1)  # [S]
    return quantile.to_py()

  def _best_y(self, feature_batch,
              aux):
    """Extract the best quantized y value from sample feature batch dict."""
    num_trials = len(self._historical_study.trials)
    if num_trials == 0:
      return 0

    f_indices = [
        self._inference_model.fun_index_in_trial(trial_index=i, aux=aux)
        for i in range(num_trials)
    ]
    # Shape of decoder_target_tokens: [S, seq_length]
    # All samples share the same history function values.
    fs = feature_batch['decoder_target_tokens'][0, f_indices]  # [#trials]

    # Convert from token index [0, V-1] to quantized integer in [0, Q-1]
    fs = fs - self._inference_model.vocab_index_from
    return int(jnp.max(fs))

  def _pi(
      self,
      logits,  # [S, V].
      feature_batch,
      aux):
    """Compute the expected improvement."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    best_y = self._best_y(feature_batch, aux)  # Quantized best y value.
    log_probs = inference_utils.logits_to_log_probs(logits)
    pis = jnp.exp(log_probs[:, best_y + 1:]).sum(1)
    return pis.to_py()  # [S]

  def _ei(
      self,
      logits,  # [S, V].
      feature_batch,
      aux):
    """Compute the expected improvement."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    best_y = self._best_y(feature_batch, aux)  # Quantized best y value.
    if best_y == logits.shape[1] - 1:
      # Highest y value is already observed.
      return np.zeros(logits.shape[0])

    # Number of bins with higher value than best_y: Q - best_y - 1.
    imp_bins = logits.shape[1] - best_y - 1

    # y - best_y, forany y in [1, imp_bins].
    imp_y_range = jnp.arange(1, imp_bins + 1)  # [1, ..., imp_bins].

    # log probabilities of improved y bins.
    log_probs = inference_utils.logits_to_log_probs(logits)
    imp_log_probs = log_probs[:, best_y + 1:]  # [S, imp_bins].

    eis = (jnp.exp(imp_log_probs) * imp_y_range[None, :]).sum(1)
    return eis.to_py()  # [S]

  def _expected_mean(
      self,
      logits  # [S, V].
  ):
    """Compute the expected mean of discrete function distributions."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    y_range = jnp.arange(logits.shape[1])
    log_probs = inference_utils.logits_to_log_probs(logits)
    mean = (jnp.exp(log_probs) * y_range[None, :]).sum(1)
    return mean.to_py()  # [S]

  def _thompson_sampling(
      self,
      fun_samples,  # [S, T-1] tokens or None
      logits,  # [S, V].
      aux,
  ):
    del aux
    # Sample the last function value.
    rng, _ = random.split(self._inference_model._rng)  # pylint:disable=protected-access
    last_funs = random.categorical(rng, logits).astype(jnp.int32).to_py()  # [S]
    if self._num_trials > 1:
      fun_samples = np.concatenate([fun_samples, last_funs[:, None]], axis=1)
    else:
      fun_samples = last_funs[:, None]
    max_funs = fun_samples.max(1)  # [S]
    return max_funs

  def Update(self, trials):
    completed_trials = []
    for trial in trials:
      if trial.status == pyvizier.Trial.COMPLETED:
        # A completed trial either has a final_measurement or is marked as
        # trial_infeasible.
        copied_trial = copy.deepcopy(trial)
        if self._metric_flipped:
          experimenters.flip_trial_metric_values(copied_trial)
        completed_trials.append(copied_trial)
    self._historical_study.trials.extend(completed_trials)  # pytype:disable=attribute-error

  def _dummy_trial(self, study):
    """Make a dummy trial."""
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

    if len(sc.metric_information) != 1:
      raise ValueError('Study contains zero or multiple metric information.')
    py_trial.complete(
        pyvizier.Measurement({list(sc.metric_information)[0].name: 1.0}))
    trial = pyvizier.TrialConverter.to_proto(py_trial)
    return trial

  def _study_to_sample(self):
    """Make a study to feed to the model."""
    study = pyvizier.StudyWithTrials(
        problem=self._historical_study.study_config)
    study.trials.extend(self._historical_study.trials)
    if not study.trials:
      # Make a dummy trial so that the converter will return non-empty string.
      study.trials.append(self._dummy_trial(study))
    return study

  def DebugString(self):
    return str({
        'study_config': self._study_config,
        'inference_model': self._inference_model
    })

  def Reset(self):
    self._historical_study = pyvizier.StudyWithTrials(
        problem=self._study_config)
    self._suggest_history = collections.defaultdict(list)

  def make_score_fn(
      self,
      num_trials):
    """Return a function to compute 1D array of scores for a list of trials."""

    # Prepare feature batch and aux.
    study = pyvizier.StudyWithTrials(
        problem=self._historical_study.study_config,
        trials=self._historical_study.trials)
    # Pad the study with a dummy trial as a placeholder for the candadiates.
    padded_study = self._inference_model.pad_study(study, len(study.trials) + 1)

    # Convert the study once and precompute the feature batch.
    _, batch_list, aux_list = self._inference_model.compute_logits(
        [padded_study], restrict_vocab_index=True)  # [S, T, V].
    aux = aux_list[0]
    trial_ids = aux.get('trial_ids') or aux['trial_permutation']['trial_ids']
    if len(trial_ids) - 1 != len(study.trials):
      raise ValueError('The number of converted trials ({len(trial_ids)}) - 1 '
                       f'does not match the input study ({len(study.trials)}). '
                       'Some trials are not valid.')

    single_example_feature = batch_list[0]
    feat_batch = {
        k: jnp.repeat(v, num_trials, axis=0)
        for k, v in single_example_feature.items()
    }

    # Find the parameter index range.
    num_parameters = self._inference_model.num_parameters(aux)
    trial_idx = len(self._historical_study.trials)
    param_index_range = self._study_converter.trial_token_scheme.param_index_range_in_trial(
        num_parameters, trial_idx)
    # Decoder input is the target shifted to the right by one.
    input_slice = slice(*(i + 1 for i in param_index_range))
    vocab_index_from = self._inference_model.vocab_index_from

    if 'type' not in self._ranking_config:
      raise ValueError('ranking_config must include a key of "type" when '
                       'making a score function.')
    ranking_type = self._ranking_config['type']
    if ranking_type == 'expected_mean':
      # Select the sample with the highest mean.
      score_fn_with_logits = self._expected_mean
    elif ranking_type == 'ucb':
      score_fn_with_logits = self._ucb
    elif ranking_type == 'thompson_sampling':
      raise NotImplementedError('thompson_sampling is not supported in '
                                'make_score_fn yet.')
    elif ranking_type == 'ei':
      score_fn_with_logits = functools.partial(
          self._ei, feature_batch=feat_batch, aux=aux)
    elif ranking_type == 'pi':
      score_fn_with_logits = functools.partial(
          self._pi, feature_batch=feat_batch, aux=aux)
    else:
      raise ValueError('Unknown ranking type: "{ranking_type}".')

    def score_fn(trials):
      """Given a sequence of trials, compute the scores."""
      # Convert the list of trials to a list of trial parameter value lists.
      pytrials = trials
      parameters_list = [
          self._study_converter.pytrial_to_parameter_values(aux, trial)
          for trial in pytrials
      ]
      if any([parameters is None for parameters in parameters_list]):
        raise ValueError('Some trial cannot be converted.')

      # Convert the parameter values to token array.
      parameters = jnp.array(parameters_list, dtype=jnp.int32)
      if parameters.ndim != 2 or parameters.shape != (num_trials,
                                                      num_parameters):
        raise ValueError('The list of trials cannot be converted '
                         'to a 2-D parameter array with shape '
                         f'{(num_trials, num_parameters)}.')
      param_tokens = parameters + vocab_index_from

      # Insert parameter values to the feature batch.
      feat_batch['decoder_input_tokens'] = feat_batch[
          'decoder_input_tokens'].at[:, input_slice].set(param_tokens)

      # Compute the scores.
      logits = self.function_logits(feat_batch, trial_idx, aux)
      scores = score_fn_with_logits(logits)
      return scores

    return score_fn

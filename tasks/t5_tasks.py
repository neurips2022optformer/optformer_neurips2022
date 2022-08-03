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

"""Language modeling on vizier trials."""
# pylint:disable=undefined-variable
import functools
import random
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import gin

from optformer_neurips2022.t5x import converters
from optformer_neurips2022.t5x import distributed_dataset
from optformer_neurips2022.t5x import preprocessors
from optformer_neurips2022.t5x import vocabularies

import seqio
import t5.data
import tensorflow as tf

from vizier.service import pyvizier as oss_pyvizier

OnePathType = Union[str, Dict[str, str]]
PathType = Union[OnePathType, Tuple[OnePathType], List[OnePathType]]

SENTENCEPIECE_MODEL_FILE = ''
MAX_INTEGER_TOKENS = 1000

VIZIER_TASKS = []
VIZIER_VOCABULARY = None
OUTPUT_FEATURES_LM = None


@gin.configurable
def get_vocabulary(sentencepiece_model_file = SENTENCEPIECE_MODEL_FILE,
                   max_integer_tokens = MAX_INTEGER_TOKENS,
                   expected_vocab_size = None):
  """Create a vocabulary with extra integer tokens."""
  extra_tokens = ['<' + str(n) + '>' for n in range(max_integer_tokens)]
  vocabulary = vocabularies.SentencePieceVocabularyWithCustomToken(
      sentencepiece_model_file, extra_tokens=extra_tokens)
  if (expected_vocab_size is not None and
      expected_vocab_size != vocabulary.vocab_size):
    raise ValueError(f'Vocabulary size ({vocabulary.vocab_size}) does not '
                     f'match the expected value ({expected_vocab_size}).')
  return vocabulary


def vocabulary_size(vocab):
  """Utility function to get the vocabulary size from a gin config file."""
  return vocab.vocab_size


def reverb_server_wrapper(
    split,
    shuffle_files = False,
    seed = None,
    dataset_fn = None,
    **kwargs,
):
  return distributed_dataset.input_fn(dataset_fn, split, shuffle_files, seed,
                                      **kwargs)


def get_dataset_split(
    split,
    shuffle_files = False,
    seed = None,
    sstable_path = '',
):
  """Return the list of dataset split files, shuffled if needed.

  Args:
    split: 'train', 'test', or 'validation'. The split to load.
    shuffle_files: whether to shuffle. This should be False.
    seed: random seed.
    sstable_path: sstable path or a dict of split paths, or a sequence of them.

  Returns:
    subset_files, a list of sstable sharded files.
  """
  if isinstance(sstable_path, tuple):
    sstable_path = list(sstable_path)
  elif not isinstance(sstable_path, list):
    sstable_path = [sstable_path]
  subset_files = []

  for one_path in sstable_path:
    subset_files.extend(
        get_one_dataset_split(split=split, sstable_path=one_path))

  if shuffle_files:
    if seed is not None:
      random.seed(seed)
    random.shuffle(subset_files)

  return subset_files


def get_one_dataset_split(split, sstable_path):
  """Return a list of sstable shard files according to the split."""
  if isinstance(sstable_path, str):
    # Legacy dataset path. Splits are defined according to the shard index.
    subset_path = sstable_path
    all_sharded_files = GenerateShardedFilenames(subset_path)

    # A tentative scheme to split the SSTable assuming the Study protos are
    # random distributed in all shards.
    # Test: shard 0
    # Validation: shard 1
    # Training: other shards.
    if split == 'train':
      subset_files = all_sharded_files[2:]
    elif split == 'validation':
      subset_files = all_sharded_files[1:2]
    elif split == 'test':
      subset_files = all_sharded_files[:1]
    else:
      raise ValueError('Unknown split %s' % split)
  else:
    # New path format. Separate sstable paths are given in a dict.
    subset_path = sstable_path[split]
    subset_files = GenerateShardedFilenames(subset_path)
  return subset_files


def build_vizier_dataset_pyfunc(
    split,
    shuffle_files = False,
    seed = None,
    sstable_path = '',
    study_converter = None,
    deterministic_map = None,
):
  """Build a tf.data Dataset of Vizier studies given split.

  Load data with tf.data.SSTableDataset and convert with numpy_function.

  Args:
    split: 'train', 'test', or 'validation'. The split to load.
    shuffle_files: whether to shuffle. This should be False.
    seed: random seed.
    sstable_path: sstable path or a dict of split paths, or a sequence of them.
    study_converter: study converter.
    deterministic_map: whether map(numpy_function, ...) is run determistically.

  Returns:
    a tf.data.Dataset containing converted Vizier Study examples.
  """
  if study_converter is None:
    raise ValueError('study_converter must not be None.')

  subset_files = get_dataset_split(split, shuffle_files, seed, sstable_path)

  ds = tf.data.SSTableDataset(subset_files)

  def converter(key, example):
    """Return a tensor with a single string or an empty tensor."""
    del key

    def _py_converter(example):
      study_proto = example

      # Some Study protos have deprecated field values and will cause a
      # ValueError in the converter.
      try:
        texts = study_converter.study_to_texts(study_proto)
      except ValueError as e:
        logging.warning(
            'Skip a Study proto (study GUID: %s, study name: %s) because of '
            'ValueError in the converter: %s.', study_proto.study_guid,
            study_proto.study_config.name, e)
        return False, '', '', '', 0
      if not (texts.inputs and texts.target_inputs and texts.targets):
        logging.warning(
            'Skip a Study proto (study GUID: %s, study name: %s) because '
            'inputs, target_inputs or targets are empty.',
            study_proto.study_guid, study_proto.study_config.name)
        return False, '', '', '', 0
      return (True, texts.inputs, texts.target_inputs, texts.targets,
              texts.num_parameters)

    valid, *data = tf.numpy_function(
        func=_py_converter,
        inp=[example],
        Tout=[bool, tf.string, tf.string, tf.string, tf.int64])
    # Tokenizer requires inputs with a known shape.
    for x in data:
      x.set_shape(())
    return [valid] + data

  def format_outputs(valid, *data):
    del valid
    inputs, target_inputs, targets, num_parameters = data
    num_parameters = tf.cast(num_parameters, tf.int32)
    return dict(
        inputs=inputs,
        target_inputs=target_inputs,
        targets=targets,
        num_parameters=num_parameters)

  ds = (
      ds.map(
          converter,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=deterministic_map)
      # Filter out invalid examples.
      .filter(lambda valid, *data: valid).map(
          format_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE))
  return ds


def vizier_dataset_from_generator(
    study_generator,
    study_converter,
    converted_study_callback = None,
    suppress_error = True):
  """Build a tf.data Dataset of Vizier studies from a generator.

  Generate study with a generator.

  Args:
    study_generator: a study generator that supports the `iter()` protocol.
    study_converter: study converter.
    converted_study_callback: a callback function with the converted study.
    suppress_error: suppress errors and log a warning message instead.

  Returns:
    a tf.data.Dataset containing converted Vizier Study examples.
  """

  def _make_gen():
    # Make generated protos.
    for study_proto in study_generator():
      # Some Study protos have deprecated field values and will cause a
      # ValueError in the converter.
      try:
        texts = study_converter.study_to_texts(study_proto)
      except ValueError as e:
        if suppress_error:
          logging.warning(
              'Skip a Study proto (study GUID: %s, study name: %s) because of '
              'ValueError in the converter: %s.', study_proto.study_guid,
              study_proto.study_config.name, e)
          continue
        else:
          raise e
      if not (texts.inputs and texts.target_inputs and texts.targets):
        if suppress_error:
          logging.warning(
              'Skip a Study proto (study GUID: %s, study name: %s) because '
              'inputs, target_inputs or targets are empty.',
              study_proto.study_guid, study_proto.study_config.name)
          continue
        else:
          raise ValueError('A Study proto cannot be converted.')

      if converted_study_callback is not None:
        converted_study_callback(texts)

      yield dict(
          inputs=texts.inputs,
          target_inputs=texts.target_inputs,
          targets=texts.targets,
          num_parameters=texts.num_parameters)

  output_types = dict(
      inputs=tf.string,
      target_inputs=tf.string,
      targets=tf.string,
      num_parameters=tf.int32)
  output_shapes = dict(
      inputs=(), target_inputs=(), targets=(), num_parameters=())
  return tf.data.Dataset.from_generator(
      generator=_make_gen,
      output_types=output_types,
      output_shapes=output_shapes)


def build_vizier_dataset_from_sstable_generator(
    split,
    shuffle_files = False,
    seed = None,
    sstable_path = '',
    study_converter = None):
  """Build a tf.data Dataset of Vizier studies given split.

  Load data from sstable and convert to tf.data dataset with a python generator.

  Args:
    split: 'train', 'test', or 'validation'. The split to load.
    shuffle_files: whether to shuffle. This should be False.
    seed: random seed.
    sstable_path: sstable path or a dict of split paths, or a sequence of them.
    study_converter: study converter.

  Returns:
    a tf.data.Dataset containing converted Vizier Study examples.
  """
  if study_converter is None:
    raise ValueError('study_converter must not be None.')

  def _generator():
    subset_files = get_dataset_split(split, shuffle_files, seed, sstable_path)
    for subset_file in subset_files:
      table = load_sstable_file(subset_file)

      # Iterate over protos.
      for example in table.itervalues():
        study_proto = example
        yield study_proto

  return vizier_dataset_from_generator(_generator, study_converter)


def build_vizier_dataset_from_generator(
    split,
    shuffle_files = False,
    seed = None,
    study_generator = None,
    study_converter = None):
  """A wrapper of vizier_dataset_from_generator with additional inputs."""
  assert split in ['train', 'validation', 'test']
  del shuffle_files
  del seed
  if study_generator is None:
    raise ValueError('study_generator must not be None.')
  if study_converter is None:
    raise ValueError('study_converter must not be None.')

  def _generator():
    while True:
      yield study_generator()

  return vizier_dataset_from_generator(_generator, study_converter)


class DatasetFromStudy(object):
  """Build a dataset to generate data from a study list.

  It adds a task that generates converted studies from `self._study_list` as a
  placehold. Every call to `dataset` will update `self._study_list` and return
  a dataset with the updated content.

  This class is useful for running inference with a list of given studies.
  """

  def __init__(self,
               add_tasks_fn,
               study_converter,
               feature_converter_cls,
               task_feature_lengths,
               batch_size,
               task_name = 'task_from_study'):
    """Construct a generator.

    Args:
      add_tasks_fn: callable to add a seqio task to the registry, with keyword
        arguments name, and dataset_fn. It is intended to be a partially
        configured method of `add_tasks`.
      study_converter: study converter.
      feature_converter_cls: feature converter class.
      task_feature_lengths: a mapping from feature name to length.
      batch_size: batch size.
      task_name: seqio task name.
    """
    self._study_converter = study_converter
    self._study_list: List[oss_pyvizier.StudyWithTrials] = []
    self._aux_list = []
    self._max_aux_size = 0
    self._batch_size = batch_size

    # Add a task and get dataset.
    add_tasks_fn(name=task_name, dataset_fn=self._dataset_fn)
    feature_converter = feature_converter_cls(pack=False)
    ds = seqio.get_dataset(
        mixture_or_task_name=task_name,
        task_feature_lengths=task_feature_lengths,
        feature_converter=feature_converter)
    ds = ds.batch(batch_size, drop_remainder=False)
    self._dataset: tf.data.Dataset = ds

  def dataset(
      self,
      study_list = None
  ):
    """Returns the TF dataset with the study list updated if specified.

    Note that a new call to `dataset` will change the content of the study list
    from previous calls.

    Args:
      study_list: list of study protocol buffers to update if not None.

    Returns:
      A TF dataset containing the list of studies.
    """
    if study_list is not None:
      self._study_list.clear()
      self._study_list.extend(list(study_list))
      self._aux_list.clear()
      self._max_aux_size = len(study_list)
    return self._dataset

  @property
  def study_converter(self):
    return self._study_converter

  @property
  def study_aux_list(self):
    return self._aux_list

  @property
  def batch_size(self):
    return self._batch_size

  def _dataset_fn(
      self,
      split,
      shuffle_files = False,
      seed = None,
  ):
    """Build a tf.data Dataset of Vizier studies given split.

    Read data from self._study_list and convert with a python generator.
    Arguments split, shuffle_files and seed are ignored.

    Args:
      split: 'train', 'test', or 'validation'. The split to load. Ignored.
      shuffle_files: whether to shuffle. This should be False. Ignored.
      seed: random seed. Ignored.

    Returns:
      a tf.data.Dataset containing converted Vizier Study examples.
    """
    del split, shuffle_files, seed

    def _generator():
      for study_proto in self._study_list:
        yield study_proto

    def _callback(texts):
      """Called after a study is converted."""
      if len(self._aux_list) <= self._max_aux_size:
        self._aux_list.append(texts.aux)

    return vizier_dataset_from_generator(
        study_generator=_generator,
        study_converter=self._study_converter,
        converted_study_callback=_callback,
        suppress_error=False)


def add_tasks(
    name = 'vizier',
    with_inputs = True,
    masked_types = None,
    num_initial_tokens = 1,
    dataset_fn = build_vizier_dataset_pyfunc,
    add_eos_in_targets = False):
  """Creates two tasks: 'vizier' and 'vizier_for_reverb'."""
  if not with_inputs:
    raise NotImplementedError('Option with_inputs = False has been deprecated.')

  masked_types = masked_types or []

  global VIZIER_VOCABULARY
  global OUTPUT_FEATURES_LM
  VIZIER_VOCABULARY = get_vocabulary()
  OUTPUT_FEATURES_LM = {
      'inputs':
          t5.data.Feature(vocabulary=VIZIER_VOCABULARY, add_eos=False),
      'targets':
          t5.data.Feature(
              vocabulary=VIZIER_VOCABULARY, add_eos=add_eos_in_targets),
      'target_inputs':
          t5.data.Feature(
              vocabulary=VIZIER_VOCABULARY, add_eos=add_eos_in_targets),
      'targets_types':
          t5.data.Feature(
              vocabulary=t5.data.PassThroughVocabulary(size=4), add_eos=False),
      'targets_masks':
          t5.data.Feature(
              vocabulary=t5.data.PassThroughVocabulary(size=2), add_eos=False),
  }

  # A wrapper to match the required positional arguments.
  # Gin configured functions have no positional arguments but the dataset_fn
  # argument of t5.data.TaskRegistry.add requires positional arguments of split,
  # shuffle_files, seed.
  def _dataset_fn(
      split,
      shuffle_files = False,
      seed = None,
  ):
    return dataset_fn(split, shuffle_files, seed)

  for reverb_wrapper in [False, True]:
    if reverb_wrapper:
      task_name = distributed_dataset.reverb_task_name(name)
    else:
      task_name = name
    VIZIER_TASKS.append(
        t5.data.TaskRegistry.add(
            name=task_name,
            task_cls=t5.data.FunctionTask,
            splits=('train', 'validation', 'test'),
            # Function which maps:
            # (split: str, shuffle_files: Bool)
            # -> tf.data.Dataset
            dataset_fn=(_dataset_fn
                        if not reverb_wrapper else functools.partial(
                            reverb_server_wrapper, dataset_fn=_dataset_fn)),
            text_preprocessor=[],
            output_features=OUTPUT_FEATURES_LM,
            token_preprocessor=[
                functools.partial(
                    preprocessors.add_targets_types,
                    num_initial_tokens=num_initial_tokens),
                functools.partial(
                    preprocessors.add_targets_masks, masked_types=masked_types)
            ],
            metric_fns=[],
            supports_caching=False,
            shuffle_buffer_size=100000,
        ))

  return VIZIER_TASKS

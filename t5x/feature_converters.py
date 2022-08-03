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

"""Feature converters to support customized loss weights."""

from typing import Mapping

from absl import logging

from optformer_neurips2022.t5x.utils import TargetType

import seqio
import tensorflow as tf


FeatureConverter = seqio.FeatureConverter


def _offset_mask_value(features):
  """Map the mask value {0, 1} -> 0, {2} -> 1 after packing."""
  d = dict(features)
  d["targets_masks"] = tf.cast(d["targets_masks"] > 1, d["targets_masks"].dtype)
  return d


def verify_feature_alignment(features,
                             name_1, name_2):
  for suffix in ["positions", "segment_ids"]:
    tf.debugging.assert_equal(
        features[f"{name_1}_{suffix}"], features[f"{name_2}_{suffix}"],
        message=f"{name_1} {suffix} does not align with {name_2}.")


def _apply_targets_masks(features,
                         converted_features
                         ):
  """Multiply targets_masks to decoder_loss_weights in converted features."""
  # Dictionary to return.
  d = dict(converted_features)

  targets_masks = features["targets_masks"]
  decoder_loss_weights = d["decoder_loss_weights"]
  d["decoder_loss_weights"] = tf.cast(decoder_loss_weights * targets_masks,
                                      dtype=decoder_loss_weights.dtype)
  return d


class VizierPrefixLMFeatureConverter(seqio.PrefixLMFeatureConverter):
  """Prefix LM feature converter with target masks."""
  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_types": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_masks": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_types": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32
  }

  def __init__(self,
               loss_on_targets_only = True,
               **kwargs):
    self._loss_on_targets_only = loss_on_targets_only
    super().__init__(**kwargs)

  def _convert_example(
      self, features):
    """Convert a Prefix LM example into an example with model features.

    Example:
    ```

    Suppose the original dataset is

    ds = [{"inputs": [9, 4, 6, 1],
           "targets": [3, 9, 1]},
           "targets_masks": [0, 1, 1]]

    Then the input features to this method (after padding) are

    features = {
                   "targets" = [9, 4, 6, 1, 3, 9, 1, 0, 0]
              "targets_masks" = [1, 1, 1, 1, 0, 1, 1, 0, 0]
              "inputs_width" = [4, 4, 4, 4, 4, 4, 4, 0, 0]
      "inputs_width_add_pos" = [5, 5, 5, 5, 5, 5, 5, 0, 0]
    }

    where "inputs_width" is length of "inputs" tiled across length dimension and
    "inputs_width_add_pos" is the same except that it has one additional
    position.

    First the parent class's _convert_example method is used to obtain the
    standard Prefix LM features. Then we update "decoder_loss_weights" with
    "targets_masks".

    decoder_loss_weights = [0, 0, 0, 0, 1, 1, 1, 0, 0]
          "targets_masks" = [1, 1, 1, 1, 0, 1, 1, 0, 0]
                    and    ---------------------------
    decoder_loss_weights = [0, 0, 0, 0, 0, 1, 1, 0, 0]

    ```

    Args:
      features: an input tf.data.Dataset to be converted.

    Returns:
      d: the converted features.
    """
    # First use the standard Prefix LM conversion.
    prefix_lm_features = super()._convert_example(features)

    if self.pack:
      verify_feature_alignment(features, "targets", "targets_types")
      verify_feature_alignment(features, "targets", "targets_masks")

    # Add target types.
    d = dict(prefix_lm_features)
    d["decoder_target_types"] = features["targets_types"]

    # Then apply the targets mask.
    logging.info("VizierPrefixLMFeatureConverter: "
                 "Applied target mask to decoder_loss_weights.")
    return _apply_targets_masks(d, prefix_lm_features)

  def _convert_features(
      self, ds,
      task_feature_lengths):
    """Convert the input dataset to an output dataset to be fed to the model.

    The "inputs" and "targets" are concatenated to form the new targets. In
    addition, the binary mask to distinguish "inputs" and "targets" token are
    concatenated as well.

    We also append "targets_masks" with ones at the beginning to align with the
    concatenated targets sequence.

    We define inputs_width to be a width (or a number of tokens) of "inputs" in
    the concatenated sequence. This method computes the width corresponding with
    and without additional position. Both of these are necessary
    `_convert_example`.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from task feature name to its length.

    Returns:
      ds: the converted dataset.
    """
    def concat_and_add_masks(features):
      inputs = features["inputs"]
      targets = features["targets"]
      targets_types = features["targets_types"]
      targets_masks = features["targets_masks"]

      tf.debugging.assert_equal(
          tf.size(targets), tf.size(targets_types),
          message="targets size does not equal targets_types size.")
      tf.debugging.assert_equal(
          tf.size(targets), tf.size(targets_masks),
          message="targets size does not equal targets_masks size.")

      # Width of the "inputs" portion in the concatenated sequence.
      width = tf.size(inputs)
      inputs_width = tf.fill([tf.size(inputs) + tf.size(targets)], width)

      # Width with an extra position to the right in the inputs mask. See
      # docstring for details.
      inputs_width_add_pos = tf.fill([tf.size(inputs) + tf.size(targets)],
                                     width + 1)

      # Pad the beginning with 1 to make the targets_masks size match the
      # concatenated targets.
      inputs_mask = tf.fill([tf.size(inputs)], 1)
      targets_masks = tf.concat([inputs_mask, features["targets_masks"]],
                                axis=-1)

      inputs_types = tf.fill([tf.size(inputs)], TargetType.OTHER.value)
      targets_types = tf.concat([inputs_types, features["targets_types"]],
                                axis=-1)

      # During pack, a value of 0 is considered as the padding token. Add 1 to
      # move the mask value set from {0, 1} to {1, 2} to avoid unexpected
      # behaviors during packing.
      targets_masks = targets_masks + 1

      return {
          "targets": tf.concat([inputs, targets], axis=-1),
          "targets_types": targets_types,
          "targets_masks": targets_masks,
          "inputs_width": inputs_width,
          "inputs_width_add_pos": inputs_width_add_pos,
      }

    ds = ds.map(
        concat_and_add_masks, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    concat_length = sum(task_feature_lengths.values())
    concat_task_feature_lengths = {
        "targets": concat_length,
        "targets_types": concat_length,
        "targets_masks": concat_length,
        "inputs_width": concat_length,
        "inputs_width_add_pos": concat_length
    }

    ds = self._pack_or_pad(ds, concat_task_feature_lengths)
    ds = ds.map(
        _offset_mask_value, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths):
    """Define the length relationship between input and output features."""
    feature_lengths = super().get_model_feature_lengths(task_feature_lengths)
    feature_lengths = dict(feature_lengths)
    feature_lengths.update({
        "decoder_target_types": feature_lengths["decoder_target_tokens"]})
    return feature_lengths


class VizierDecoderFeatureConverter(seqio.DecoderFeatureConverter):
  """Decoder feature converter for the Vizier project."""

  def __init__(self,
               loss_on_targets_only = True,
               pack = True,
               use_custom_packing_ops = False):

    self._loss_on_targets_only = loss_on_targets_only
    super().__init__(loss_on_targets_only=loss_on_targets_only,
                     pack=pack, use_custom_packing_ops=use_custom_packing_ops)
    # Override the prefixlm_feature_converter.
    self.prefixlm_feature_converter = VizierPrefixLMFeatureConverter(
        loss_on_targets_only=loss_on_targets_only,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops)


class VizierEncDecFeatureConverter(seqio.EncDecFeatureConverter):
  """Prefix encoder-decoder feature converter with target masks."""

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "target_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_types": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_masks": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_types": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "encoder_segment_ids": tf.int32,
      "decoder_segment_ids": tf.int32,
      "encoder_positions": tf.int32,
      "decoder_positions": tf.int32
  }

  def _prepare_targets_masks(
      self, features):
    """Adjust targets_masks value before packing."""
    targets = features["targets"]
    target_inputs = features["target_inputs"]
    targets_types = features["targets_types"]
    targets_masks = features["targets_masks"]

    tf.debugging.assert_equal(
        tf.size(targets), tf.size(target_inputs),
        message="targets size does not equal target_inputs size.")
    tf.debugging.assert_equal(
        tf.size(targets), tf.size(targets_types),
        message="targets size does not equal targets_types size.")
    tf.debugging.assert_equal(
        tf.size(targets), tf.size(targets_masks),
        message="targets size does not equal targets_masks size.")

    # During pack, a value of 0 is considered as the padding token. Add 1 to
    # move the mask value set from {0, 1} to {1, 2} to avoid unexpected
    # behaviors during packing.
    targets_masks = targets_masks + 1

    d = dict(features)
    d["targets_masks"] = targets_masks
    return d

  def _convert_example(
      self, features):
    # The first step is copied from parent class's convert_example method. We
    # cannot call the method in the parent class because it is defined inside
    # its _convert_features method.

    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = seqio.make_autoregressive_inputs(
        features["target_inputs"],
        sequence_id=features.get("targets_segment_ids", None))

    d = {"encoder_input_tokens": features["inputs"],
         "decoder_target_tokens": features["targets"],
         "decoder_target_types": features["targets_types"],
         "decoder_input_tokens": decoder_input_tokens,
         # Loss is computed for all but the padding positions.
         "decoder_loss_weights": seqio.non_padding_position(
             features["targets"])}

    if self.pack:
      d["encoder_segment_ids"] = features["inputs_segment_ids"]
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["encoder_positions"] = features["inputs_positions"]
      d["decoder_positions"] = features["targets_positions"]

      verify_feature_alignment(features, "targets", "target_inputs")
      verify_feature_alignment(features, "targets", "targets_types")
      verify_feature_alignment(features, "targets", "targets_masks")

    # Second step: multiply targets mask with decoder_loss_weights.
    logging.info("VizierEncDecFeatureConverter: "
                 "Applied target mask to decoder_loss_weights.")
    return _apply_targets_masks(features, d)

  def _convert_features(
      self, ds,
      task_feature_lengths):
    """Convert the dataset to be fed to the encoder-decoder model.

    Compared to its parent class seqio.EncDecFeatureConverter, it multiplies
    the targets mask to the created decoder_loss_weights sequence.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """

    ds = ds.map(self._prepare_targets_masks,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = self._pack_or_pad(ds, task_feature_lengths)
    ds = ds.map(
        _offset_mask_value, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths):
    """Define the length relationship between input and output features."""
    feature_lengths = super().get_model_feature_lengths(task_feature_lengths)
    feature_lengths = dict(feature_lengths)
    feature_lengths.update({
        "decoder_target_types": feature_lengths["decoder_target_tokens"]})
    return feature_lengths

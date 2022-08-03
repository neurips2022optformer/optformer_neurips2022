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

"""GP-UCB."""
# pylint:disable=undefined-variable
import dataclasses
import functools
from typing import Any, Callable, List, Optional, Sequence

from absl import logging
import attr
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from vizier import algorithms as vza
from vizier import keras as vzk
from vizier import pyvizier as vz
from vizier import tfp as vzp
from vizier.pyvizier import converters

Array = Any
tfd = tfp.distributions
tfpb = tfp.bijectors


class InverseClipping(tfpb.Identity):

  def __init__(self, minimum, name = 'inverse_clipping'):
    super().__init__(name=name)
    self._minimum = minimum

  def _inverse(self, y):
    y = tf.maximum(y, tf.convert_to_tensor(self._minimum))
    return y


@attr.define(auto_attribs=True)
class _GaussianProcessModelConfig:
  """Internal class to reduce clutter. See GaussianProcessModel.__init__."""
  num_components: int
  lbfgs_iterations: int
  lbfgs_verbose: int
  lbfgs_include_layer_metrics: bool
  lbfgs_jit_compile_train_loss: bool
  predict_joint: bool
  loss: Optional[str]
  y_transform: str = attr.field(
      validator=attr.validators.in_(['yeo-johnson', 'box-cox', 'identity']))
  dtype: str


@dataclasses.dataclass
class _GaussianProcessModelUpdateTrace:
  """Internal class, for interactive debugging purposes only."""
  gp: Optional[vzk.layers.GaussianProcessLayer] = None
  feature_transform: Optional[vzk.layers.Layer] = None
  index_points: Optional[Array] = None
  labels: Optional[Array] = None
  batch_shape: Optional[Any] = None
  optimizer: Optional[Any] = None
  results: Optional[Any] = None
  labels_transform: Optional[tfpb.Bijector] = None


class GaussianProcessModel:
  """GP Model that supports update-predict of Vizier Trials."""

  def __init__(self,
               study_config,
               *,
               num_components = 10,
               lbfgs_iterations = 200,
               lbfgs_verbose = 0,
               lbfgs_include_layer_metrics = False,
               lbfgs_jit_compile_train_loss = False,
               predict_joint = False,
               loss = 'avgnll',
               y_transform = 'yeo-johnson',
               dtype = 'float64'):
    """Init.

    Args:
      study_config: study config
      num_components: Number of GPs to train in parallel. Currently the policy
        uses the best trained model and discards the rest.
      lbfgs_iterations: Number of LBFGS iterations for optimizing
        hyperparameters.
      lbfgs_verbose: Verbosity of LBFGS.
      lbfgs_include_layer_metrics: Decides whether to include all layer metrics
        (debugging metrics) in the train log. Turning it off may improve speed.
      lbfgs_jit_compile_train_loss: Jit compile the train loss. May improve
        speed for big models or a large number of lbfgs iterations.
      predict_joint: If True, predict the joint distribution over labels.
      loss: Decides the loss function. See `GaussianProcessLayer.metrics_fn`.
      y_transform: Valid values are ('yeo-johnson', 'box-cox', 'identity').
      dtype: dtype, in string.
    """
    self._config = _GaussianProcessModelConfig(num_components, lbfgs_iterations,
                                               lbfgs_verbose,
                                               lbfgs_include_layer_metrics,
                                               lbfgs_jit_compile_train_loss,
                                               predict_joint, loss, y_transform,
                                               dtype)
    self._incorporated_trials: List[vz.Trial] = []
    feature_converter_factory = functools.partial(
        converters.DefaultModelInputConverter,
        scale=True,
        onehot_embed=True,
        float_dtype=self.dtype)
    feature_converters = [
        feature_converter_factory(pc)
        for pc in study_config.search_space.parameters
    ]
    metric_converters = [
        converters.DefaultModelOutputConverter(mi, dtype=self.dtype)
        for mi in study_config.metric_information
    ]
    self.num_metrics = len(metric_converters)
    if self.num_metrics > 1:
      self._config.num_components = 0
      logging.warning(
          'Number of components must be 0 for multi-metric'
          'with %d metrics', self.num_metrics)

    self._converter = converters.DefaultTrialConverter(feature_converters,
                                                       metric_converters)

    # Call update() to initialize the predictor states.
    self.update([])

  def _create_model(self):
    """Creates GP."""
    if self._config.num_components:
      batch_shape = [self._config.num_components, self.num_metrics]
    else:
      batch_shape = [self.num_metrics]

    # Scaling is applied at converter level.
    # Use one-hot embedding.
    embedding = vzk.layers.VizierEmbeddingLayer.from_converter(
        self._converter,
        trainable=False,
        apply_scaling=False,
        dtype=self._config.dtype)
    # Define non-linear transformation
    constraint = tfpb.SoftClip(
        tf.constant(1e-2, self._config.dtype),
        tf.constant(10., self._config.dtype),
        hinge_softness=tf.constant(1e-1, self._config.dtype))
    kumaraswamy = vzk.layers.KumaraswamyTransformationLayer(
        vzk.layers.variable_from_prior(
            tfd.LogNormal(
                tf.constant(.0, self._config.dtype),
                tf.constant(.75, self._config.dtype)),
            bijector=constraint,
            sample_constraint=constraint,
            ndims=1,
            name='concentration1',
            dtype=self._config.dtype),
        vzk.layers.variable_from_prior(
            tfd.LogNormal(
                tf.constant(.0, self._config.dtype),
                tf.constant(.75, self._config.dtype)),
            bijector=constraint,
            sample_constraint=constraint,
            ndims=1,
            name='concentration0',
            dtype=self._config.dtype),
        batch_shape=batch_shape,
        dtype=self._config.dtype)
    feature_transform_layer = vzk.layers.CompositeLayer(
        [embedding, kumaraswamy], dtype=self._config.dtype)

    # Build gp.
    gp = vzk.layers.GaussianProcessLayer(
        vzk.layers.ScaledLinearKernelLayer(
            batch_shape=batch_shape, trainable=False, dtype=self._config.dtype)
        + vzk.layers.ScaledMaternKernelLayer(
            1.5,
            batch_shape=batch_shape,
            name='matern',
            dtype=self._config.dtype),
        batch_shape=batch_shape,
        feature_transform_layer=feature_transform_layer,
        name='gp',
        dtype=self._config.dtype)
    gp.build(self._converter.features_shape)
    return gp

  def _set_prior_as_predictive(self):
    """Initialize the predictor to use the unconditioned GP.

    This method is called in the absence of observed data.
    """
    gp = self._create_model()
    # When GP is empty, choose any index to match the shape.
    self._trace.gp = gp
    if self._config.num_components:
      gp.batch_index = 0
    self._label_transform = tfpb.Identity()
    self._gp = gp
    self._predictor = gp.as_predictor(joint=self._config.predict_joint)
    return

  def update(self, trials):
    """Update the new trials."""
    self._trace = _GaussianProcessModelUpdateTrace()
    self._transformed_ys = tf.zeros([0, self.num_metrics], dtype=self.dtype)
    self._incorporated_trials.extend(trials)
    if not self._incorporated_trials:
      return self._set_prior_as_predictive()

    # Process Trials into features
    xs = self._converter.to_features(self._incorporated_trials)
    ys = converters.dict_to_array(
        self._converter.to_labels(self._incorporated_trials))

    if self._config.y_transform == 'identity':
      self._label_transform = tfpb.Identity()
    else:
      self._label_transform = vzp.bijectors.optimal_power_transformation(
          ys, self._config.y_transform)
      # Add an inverse clipping layer to avoid NaN.
      minimum = (self._label_transform(-float('inf')).numpy()[0] +
                 tf.keras.backend.epsilon())
      self._label_transform = tfp.bijectors.Chain([InverseClipping(minimum),
                                                   self._label_transform])

    self._transformed_ys = self._label_transform(ys).numpy()
    ys = self._transformed_ys.T  # GP expects the last dimension for metrics.
    self._trace.index_points = xs
    self._trace.labels = ys

    if not np.all(np.isfinite(ys)):
      logging.error('Preprocessig failed!')
      return self._set_prior_as_predictive()

    # Optimize GP hyperparameters.
    gp = self._create_model()
    self._trace.gp = gp
    optimizer = vzk.optim.LbfgsOptimizer(
        batch_ndims=1, f_absolute_tolerance=1e-2)
    self._trace.optimizer = optimizer

    def metrics_fn(xs, ys):
      # TODO: Check if HEBO implementation uses avgnll or nll.
      metrics = gp.metrics_fn(
          xs,
          ys,
          loss=self._config.loss,
          include_layer_metrics=self._config.lbfgs_include_layer_metrics)
      # Last axis is the metric index. Sum over them.
      metrics['loss'] = tf.reduce_sum(metrics['loss'], axis=-1)
      return metrics

    logging.info('Metrics before training: %s', metrics_fn(xs, ys))

    results = optimizer.optimize(
        functools.partial(metrics_fn, xs, ys),
        gp.trainable_variables,
        iterations=self._config.lbfgs_iterations,
        verbose=self._config.lbfgs_verbose,
        jit_compile=self._config.lbfgs_jit_compile_train_loss)
    post_train_metrics = metrics_fn(xs, ys)
    logging.info(
        'Training finished. Metrics after training: %s. '
        'Model components for debugging purposes are saved in '
        '`self._trace`.', post_train_metrics)

    self._trace.results = results

    if self._config.num_components:
      # Take the best GP only.
      gp.batch_index = np.nanargmin(post_train_metrics['avgnll'].numpy())
    self._gp = gp

    # Pre-compute cholesky and define acquisition.
    self._predictor = gp.predictor(xs, ys, joint=self._config.predict_joint)

  def predict(self, trials):
    """Predicts in the original label space."""
    return self.predict_from_array(self._converter.to_features(trials))

  def predict_from_array(self, xs):
    """Predicts in the original label space from feature arrays."""
    return tfpb.Invert(self._label_transform)(
        self.predict_transformed_label_from_array(xs))

  def predict_transformed_label_from_array(self, xs):
    """Predicts in the transformed label space."""
    # xs are shape [batch size, feature_dim].
    if self.num_metrics <= 1:
      return self._predictor(xs)
    else:
      predictor_output = self._predictor(xs)
      # Push all dims from batch to event shape and transpose.
      prediction = tfd.Independent(
          predictor_output,
          reinterpreted_batch_ndims=len(predictor_output.batch_shape))
      transpose = tfp.bijectors.Transpose(
          rightmost_transposed_ndims=len(prediction.event_shape))
      return transpose(prediction)

  @property
  def transformed_ys(self):
    """Returns incorporated ys in the transformed label space.

    Transformed ys shape is [num_trials, num_metrics].
    """
    return self._transformed_ys

  @property
  def converter(self):
    return self._converter

  @property
  def gp(self):
    return self._gp

  @property
  def dtype(self):
    return self._config.dtype


@attr.define
class _GaussianProcessBanditConfig:
  """Internal class to reduce clutter. See GaussianProcessBandit.__init__."""
  num_candidates: int = attr.field()
  ehv_num_vectors: int = attr.field()
  ehv_num_samples: int = attr.field()
  jit_compile_acquisitions: bool = attr.field()


@attr.define(init=False)
class GaussianProcessBandit(vza.Designer):
  """GP-bandit algorithm.

  It is GP-UCB (https://arxiv.org/abs/0912.3995) except that the UCB
  coefficient is fixed at a constant value.
  """
  _study_config: vz.StudyConfig = attr.field()
  model: GaussianProcessModel = attr.field()
  _config: _GaussianProcessBanditConfig = attr.field()
  _incorporated_trials: list[vz.Trial] = attr.field(
      factory=list, init=False, repr=lambda x: '{len(x)} trials.')

  # flip_sign_bijector: tfp.
  def __init__(
      self,
      study_config,
      model_factory = GaussianProcessModel,
      *,
      num_candidates = 15000,
      ehv_num_vectors = 100,
      ehv_num_samples = 50,
      jit_compile_acquisitions = False,
  ):
    """Init.

    Args:
      study_config: Study config.
      model_factory: Gaussian process model factory.
      num_candidates: Number of candidates to evaluate acquisition function.
      ehv_num_vectors: See `ExpectedScalarizedHyperVolume`.
      ehv_num_samples: See `ExpectedScalarizedHyperVolume`.
      jit_compile_acquisitions: If True, jit compile the acquisition function.
    """
    self.__attrs_init__(
        study_config, model_factory(study_config),
        _GaussianProcessBanditConfig(num_candidates, ehv_num_vectors,
                                     ehv_num_samples, jit_compile_acquisitions))

  def __attrs_post_init__(self):
    # Bijector to flip signs so that every metric MAXIMIZEs.
    self.flip_sign_bijector = vzp.bijectors.flip_sign(
        np.array([
            mi.goal == vz.ObjectiveMetricGoal.MINIMIZE
            for mi in self._study_config.metric_information
        ]),
        dtype=self.model.dtype)

  def update(self, delta):
    """Update.

    Args:
      delta: Completed trials only.
    """
    self._incorporated_trials.extend(delta.completed)
    self.model.update(delta.completed)

  def predict(self, trials):
    self.model.predict(trials)

  def suggest(self,
              count = None):
    """Suggest up to `count` number of trials."""
    count = count or 1
    jit_compile = self._config.jit_compile_acquisitions

    # Define and JIT compile the acquisition which takes batched candidates
    # of shape [batch size, feature_dim, num_metrics] and outputs [batch size].
    if self._study_config.metric_information.is_single_objective:

      @tf.function(jit_compile=jit_compile)
      def _array_to_acquisition(candidates):
        posterior = self.flip_sign_bijector(
            self.model.predict_transformed_label_from_array(candidates))
        return vzp.acquisitions.UpperConfidenceBound(
            tf.constant(1.8, self.model.dtype)).evaluate(posterior)
    else:
      # Apply EHVI in transformed space (with sign flips).
      ys = self.flip_sign_bijector(self.model.transformed_ys)

      @tf.function(jit_compile=jit_compile)
      def _array_to_acquisition(candidates):
        post = self.model.predict_transformed_label_from_array(candidates)
        posterior = self.flip_sign_bijector(post)
        return vzp.acquisitions.ExpectedScalarizedHypervolume(
            points=ys,
            origin=tf.reduce_min(ys, axis=0),
            num_vectors=self._config.ehv_num_vectors,
            num_samples=self._config.ehv_num_samples).evaluate(posterior)

    def _acquisition(trials):
      features = self.model.converter.to_features(trials)
      return _array_to_acquisition(features).numpy()

    # Optimize acquisition function.
    suggestions = optimizer.maximize(
        _acquisition,
        self._study_config.search_space,
        count=count)
    return [
        vz.TrialSuggestion(s.parameters, metadata=s.metadata)
        for s in suggestions
    ]

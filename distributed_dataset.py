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

"""Distributed dataset based on courier.

Assume the dataset_fn returns a single data item at a time without the batch
dimension.
"""

import time
from typing import Optional

from absl import flags
from absl import logging
import gin
import reverb
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tree

REVERB_TABLE_NAME = 'data'

# Reverb configuration for client. Used for producer and input_fn.
flags.DEFINE_string('reverb_address', None, 'host:port where reverb is running')

# Reverb port. Used for server.
flags.DEFINE_integer('reverb_port', None, 'port to run the reverb server')

# Producer task replica id used to set different seed for each producer.
flags.DEFINE_integer('taskid', None, 'Data producer task id.')

FLAGS = flags.FLAGS


@gin.configurable
def input_fn(
    dataset_fn,
    split,
    shuffle_files,
    seed,
    producer_split,
    producer_shuffle_files,
    num_workers_per_iterator,
    max_samples_per_stream,
    num_parallel_calls,
    prefetch):
  """Construct a courier dataset expecting samples from dataset_fn."""

  # Make sure the dataset arguments passed by the trainer match those in the
  # producer.
  assert (split == producer_split and
          shuffle_files == producer_shuffle_files)

  # Construct a dummy data pipeline to know the shapes to expect.
  ds = dataset_fn(split, shuffle_files, seed)
  shapes = tf.data.get_output_shapes(ds)
  dtypes = tf.data.get_output_types(ds)

  def _input_fn():
    """Input fn receiving data from reverb."""
    ds = reverb.TimestepDataset(
        server_address=FLAGS.reverb_address,
        table=REVERB_TABLE_NAME,
        dtypes=dtypes,
        shapes=shapes,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples_per_stream=max_samples_per_stream,
        max_in_flight_samples_per_worker=2
    )
    if isinstance(dtypes, dict):
      input_keys = dtypes.keys()
      ds = ds.map(lambda x: {key: x.data[key] for key in input_keys},
                  num_parallel_calls=num_parallel_calls)
    elif isinstance(shapes, tuple):
      ds = ds.map(lambda x: x.data,
                  num_parallel_calls=num_parallel_calls)
    if prefetch:
      ds = ds.prefetch(prefetch)

    # Configure optimization options.
    optimization_options = tf.data.experimental.OptimizationOptions()
    optimization_options.apply_default_optimizations = True
    optimization_options.map_and_batch_fusion = True
    optimization_options.parallel_batch = True
    options = tf.data.Options()
    options.experimental_optimization = optimization_options
    return ds.with_options(options)

  return _input_fn()


@gin.configurable
def produce(dataset_fn,
            split,
            shuffle_files,
            seed):
  """Sends data items to a reverb client."""
  if seed is None:
    seed = int(time.time())
  if FLAGS.taskid is not None:
    seed += FLAGS.taskid * 100

  reverb_address = FLAGS.reverb_address
  reverb_client = reverb.Client(reverb_address)

  ds = dataset_fn(split, shuffle_files, seed)
  ds = ds.repeat()

  ds = iter(tfds.as_numpy(ds))

  count = 0
  for data in ds:
    logging.log_every_n(logging.INFO, 'Adding an item %d', 1000, count)
    count += 1
    # data_v is a list of tensors
    data_v = tree.flatten(data)

    # Here we are sending one element at a time. Alternatively, we can send
    # the whole batch, but the training job will need to know batch size
    # used on the data producer side.
    row = data_v
    try:
      reverb_client.insert(row, {REVERB_TABLE_NAME: 1.})
    except RuntimeError as e:
      if str(e) == 'Error when confirming that all items written to table.':
        # This can happen when servers go down. More specifically when
        # "The writer has sent items that should be inserted and the server
        # may or may not have completed the request but was unable to send the
        # confirmation that the job has been completed before the connection
        # was lost", according to cassirer@.
        #
        # As a workaround, we just ignore the error and try again with the
        # next item. This may result in some data being lost. We don't know
        # whether or not the server received the data, so I'm not going to
        # retry as I'm judging that usually it'd be worse to risk introducing
        # duplicates than to drop some data.
        #
        # Longer term, the recommended approach is to switch to use reverb's
        # trajectory_writer, which doesn't suffer from this issue and should
        # perform better too, but this requires more work.
        logging.warning(
            'Reverb client insert failed (%s). This can happen if reverb '
            'servers go down. Ignoring and trying again with the next item.',
            e)
      else:
        raise


@gin.configurable
def run_reverb_server(reverb_buffer_size):
  """Run."""

  server = reverb.Server(tables=[
      reverb.Table(
          name=REVERB_TABLE_NAME,
          sampler=reverb.selectors.Fifo(),
          remover=reverb.selectors.Fifo(),
          max_size=reverb_buffer_size,
          rate_limiter=reverb.rate_limiters.MinSize(1),
          max_times_sampled=1),
      ], port=FLAGS.reverb_port)

  server.wait()


@gin.configurable
def multiply(x, y):
  return x * y


def reverb_task_name(name):
  return name + '_for_reverb'

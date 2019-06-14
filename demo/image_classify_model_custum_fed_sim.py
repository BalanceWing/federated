#####################################################################
#
#  created by chilie.tan 
#  version 1.0 
#  2019.6.14
#####################################################################

#load packages 
from __future__ import absolute_import, division, print_function

import collections

from six.moves import range
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

tff.federated_computation(lambda: 'Hello, World!')()

#input data preparing
#@test {"output": "ignore"}
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

#preprocess images and define hyper-parameter
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500


def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

#sample 
#@test {"output": "ignore"}
preprocessed_example_dataset = preprocess(example_dataset)

#sample_batch = tf.nest.map_structure(
sample_batch = tf.contrib.framework.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())


def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

#@test {"output": "ignore"}
NUM_CLIENTS = 3

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

##########################################################
#model created with keras: compile and excute
'''
def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
  def loss_fn (y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
  
  model.compile(
      #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      #optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
      loss=loss_fn,
      optimizer=gradient_descent.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)
'''

#customizing the model implementation
MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

def create_mnist_variables():
  return MnistVariables(
      weights = tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias = tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples = tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum = tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum = tf.Variable(0.0, name='accuracy_sum', trainable=False))

def mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(tf.reduce_sum(
      tf.one_hot(flat_labels, 10) * tf.log(y), reduction_indices=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.to_float(tf.size(batch['y']))

  tf.assign_add(variables.num_examples, num_examples)
  tf.assign_add(variables.loss_sum, loss * num_examples)
  tf.assign_add(variables.accuracy_sum, accuracy * num_examples)

  return loss, predictions

def get_local_mnist_metrics(variables):
  return collections.OrderedDict([
      ('num_examples', variables.num_examples),
      ('loss', variables.loss_sum / variables.num_examples),
      ('accuracy', variables.accuracy_sum / variables.num_examples)
    ])

@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return {
      'num_examples': tff.federated_sum(metrics.num_examples),
      'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
      'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)
  }

#constructing an instance of tff model
class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                        tf.float32)),
                                    ('y', tf.TensorSpec([None, 1], tf.int32))])

  @tf.contrib.eager.function(autograph=False)
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    return tff.learning.BatchOutput(loss=loss, predictions=predictions)

  @tf.contrib.eager.function(autograph=False)
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients

class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

  @tf.contrib.eager.function(autograph=False)
  def train_on_batch(self, batch):
    output = self.forward_pass(batch)
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    optimizer.minimize(output.loss, var_list=self.trainable_variables)
    return output

###########################################################
#federate training with model & data
#@test {"output": "ignore"}
iterative_process = tff.learning.build_federated_averaging_process(MnistTrainableModel)

state = iterative_process.initialize()

#@test {"skip": true}
for round_num in range(1, 11):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

###########################################################
#evaluation
evaluation = tff.learning.build_federated_evaluation(MnistModel)

#@test {"output": "ignore"}
train_metrics = evaluation(state.model, federated_train_data)

print('[evaluate] training metric: {}'.format(train_metrics))

federated_test_data = make_federated_data(emnist_test, sample_clients)

#@test {"output": "ignore"}
test_metrics = evaluation(state.model, federated_test_data)

print('[evaluate] testing metric: {}'.format(test_metrics))

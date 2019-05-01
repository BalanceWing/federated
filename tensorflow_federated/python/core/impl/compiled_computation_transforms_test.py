# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for compiled_computation_transforms.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import compiled_computation_transforms
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl


def _make_building_block_executable(comp):
  return computation_impl.ComputationImpl(comp.proto,
                                          context_stack_impl.context_stack)


class CompiledComputationUtilsTest(test.TestCase):

  def test_select_output_with_none_comp_raises_type_error(self):
    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_output(0, None)

  def test_select_output_with_none_index_raises_type_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_output(None, foo_as_building_block)

  def test_select_output_with_wrong_return_type_raises_type_error(self):
    computation_arg_type = computation_types.to_type(tf.int32)

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    with self.assertRaises(TypeError):
      compiled_computation_transforms.select_output(0, foo_as_building_block)

  def test_select_output_by_name_bad_name_raises_value_error(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    with self.assertRaises(ValueError):
      compiled_computation_transforms.select_output('x', foo_as_building_block)

  def test_select_output_by_index_single_level_of_nesting(self):
    computation_arg_type = computation_types.NamedTupleType(
        [tf.int32, tf.float32])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    first_element_selected = compiled_computation_transforms.select_output(
        0, foo_as_building_block)
    self.assertEqual(first_element_selected.type_signature.result,
                     computation_types.to_type(tf.int32))

    second_element_selected = compiled_computation_transforms.select_output(
        1, foo_as_building_block)
    self.assertEqual(second_element_selected.type_signature.result,
                     computation_types.to_type(tf.float32))
    self.assertEqual(
        _make_building_block_executable(first_element_selected)([1, 2.]), 1)
    self.assertEqual(
        _make_building_block_executable(second_element_selected)([1, 2.]), 2.)

  def test_select_output_by_name_single_level_of_nesting(self):
    computation_arg_type = computation_types.NamedTupleType([('a', tf.int32),
                                                             ('b', tf.float32)])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    first_element_selected = compiled_computation_transforms.select_output(
        'a', foo_as_building_block)
    self.assertEqual(first_element_selected.type_signature.result,
                     computation_types.to_type(tf.int32))

    second_element_selected = compiled_computation_transforms.select_output(
        'b', foo_as_building_block)
    self.assertEqual(second_element_selected.type_signature.result,
                     computation_types.to_type(tf.float32))
    self.assertEqual(
        _make_building_block_executable(first_element_selected)({
            'a': 1,
            'b': 2.
        }), 1)
    self.assertEqual(
        _make_building_block_executable(second_element_selected)({
            'a': 1,
            'b': 2.
        }), 2.)

  def test_select_output_by_index_two_nested_levels_keeps_nested_type(self):
    nested_type1 = computation_types.NamedTupleType([('a', tf.int32),
                                                     ('b', tf.float32)])
    nested_type2 = computation_types.NamedTupleType([('c', tf.int32),
                                                     ('d', tf.float32)])

    computation_arg_type = computation_types.NamedTupleType([
        ('x', nested_type1), ('y', nested_type2)
    ])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    first_element_selected = compiled_computation_transforms.select_output(
        0, foo_as_building_block)
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_output(
        1, foo_as_building_block)
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)
    nested_elem1 = {'a': 1, 'b': 2.}
    nested_elem2 = {'c': 3, 'd': 4.}
    self.assertEqual(
        _make_building_block_executable(first_element_selected)({
            'x': nested_elem1,
            'y': nested_elem2
        }), anonymous_tuple.from_container(nested_elem1))
    self.assertEqual(
        _make_building_block_executable(second_element_selected)({
            'x': nested_elem1,
            'y': nested_elem2
        }), anonymous_tuple.from_container(nested_elem2))

  def test_select_output_by_name_two_nested_levels_keeps_nested_type(self):
    nested_type1 = computation_types.NamedTupleType([('a', tf.int32),
                                                     ('b', tf.float32)])
    nested_type2 = computation_types.NamedTupleType([('c', tf.int32),
                                                     ('d', tf.float32)])

    computation_arg_type = computation_types.NamedTupleType([
        ('x', nested_type1), ('y', nested_type2)
    ])

    @computations.tf_computation(computation_arg_type)
    def foo(x):
      return x

    foo_as_building_block = computation_building_blocks.CompiledComputation(
        computation_impl.ComputationImpl.get_proto(foo))

    first_element_selected = compiled_computation_transforms.select_output(
        'x', foo_as_building_block)
    self.assertEqual(first_element_selected.type_signature.result, nested_type1)

    second_element_selected = compiled_computation_transforms.select_output(
        'y', foo_as_building_block)
    self.assertEqual(second_element_selected.type_signature.result,
                     nested_type2)
    nested_elem1 = {'a': 1, 'b': 2.}
    nested_elem2 = {'c': 3, 'd': 4.}
    self.assertEqual(
        _make_building_block_executable(first_element_selected)({
            'x': nested_elem1,
            'y': nested_elem2
        }), anonymous_tuple.from_container(nested_elem1))
    self.assertEqual(
        _make_building_block_executable(second_element_selected)({
            'x': nested_elem1,
            'y': nested_elem2
        }), anonymous_tuple.from_container(nested_elem2))


if __name__ == '__main__':
  test.main()

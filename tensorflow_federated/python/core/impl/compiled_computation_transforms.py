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
"""Holds library of transformations for on compiled computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import type_serialization


def select_output(output, comp):
  """Makes `CompiledComputation` with same input as `comp` and output `output`.

  Given an instance of `computation_building_blocks.CompiledComputation` `comp`
  with type signature (T -> <U, ...,V>),
  `select_output` returns a `CompiledComputation` representing the logic of
  calling `comp` and then selecting `output` from the
  resulting `tuple`, where `output` can be an index or a string representing the
  name of the desired field.

  Args:
    output: Instance of `str` or `int`, the name or index of the field to select
      from the output of `comp`.
    comp: Instance of `computation_building_blocks.CompiledComputation` which
      must have result type `computation_types.NamedTupleType`, the function
      from which to select `output`.

  Returns:
    An instance of `computation_building_blocks.CompiledComputation` as
    described, the result of selecting the appropriate `output` from `comp`.
  """
  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
  py_typecheck.check_type(output, (str, int))
  proto = comp.proto
  graph_result_binding = proto.tensorflow.result
  binding_oneof = graph_result_binding.WhichOneof('binding')
  if binding_oneof != 'tuple':
    raise TypeError(
        'Can only select output from a CompiledComputation with return type '
        'tuple; you have attempted a selection from a CompiledComputation '
        'with return type {}'.format(binding_oneof))
  proto_type = type_serialization.deserialize_type(proto.type)
  if isinstance(output, int):
    result = [x for x in graph_result_binding.tuple.element][output]
    result_type = proto_type.result[output]
  else:
    type_names_list = [
        x[0] for x in anonymous_tuple.to_elements(proto_type.result)
    ]
    index = type_names_list.index(output)
    result = [x for x in graph_result_binding.tuple.element][index]
    result_type = proto_type.result[index]
  serialized_type = type_serialization.serialize_type(
      computation_types.FunctionType(proto_type.parameter, result_type))
  selected_comp = pb.Computation(
      type=serialized_type,
      tensorflow=pb.TensorFlow(
          graph_def=proto.tensorflow.graph_def,
          initialize_op=proto.tensorflow.initialize_op,
          parameter=proto.tensorflow.parameter,
          result=result))
  return computation_building_blocks.CompiledComputation(selected_comp)

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: termination_handler_params_msg.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$termination_handler_params_msg.proto\"l\n ContactTerminationHandlerMessage\x12\x13\n\x0b\x62uffer_time\x18\x01 \x02(\x01\x12\x18\n\x10\x66orce_thresholds\x18\x02 \x03(\x01\x12\x19\n\x11torque_thresholds\x18\x03 \x03(\x01\"F\n\x15JointThresholdMessage\x12\x13\n\x0b\x62uffer_time\x18\x01 \x02(\x01\x12\x18\n\x10joint_thresholds\x18\x02 \x03(\x01\"h\n\x14PoseThresholdMessage\x12\x13\n\x0b\x62uffer_time\x18\x01 \x02(\x01\x12\x1b\n\x13position_thresholds\x18\x02 \x03(\x01\x12\x1e\n\x16orientation_thresholds\x18\x03 \x03(\x01\"4\n\x1dTimeTerminationHandlerMessage\x12\x13\n\x0b\x62uffer_time\x18\x01 \x02(\x01')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'termination_handler_params_msg_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CONTACTTERMINATIONHANDLERMESSAGE._serialized_start=40
  _CONTACTTERMINATIONHANDLERMESSAGE._serialized_end=148
  _JOINTTHRESHOLDMESSAGE._serialized_start=150
  _JOINTTHRESHOLDMESSAGE._serialized_end=220
  _POSETHRESHOLDMESSAGE._serialized_start=222
  _POSETHRESHOLDMESSAGE._serialized_end=326
  _TIMETERMINATIONHANDLERMESSAGE._serialized_start=328
  _TIMETERMINATIONHANDLERMESSAGE._serialized_end=380
# @@protoc_insertion_point(module_scope)

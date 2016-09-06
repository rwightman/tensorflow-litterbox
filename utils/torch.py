# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
import torchfile
import argparse


def convert_spatial_batch_normalization(obj):
    return ""


def convert_spatial_convolution(obj):
    weights = obj.weight
    #biases = obj.biase
    kernel_width = obj.kW
    kernel_height = obj.kH
    stride_width = obj.dH
    stride_height = obj.dW
    pad_width = obj.padW
    pad_height = obj.padH
    out = '[%d, %d], %d, %s' % (kernel_height, kernel_width, stride_height, str(weights.shape))
    return out


def convert_linear(obj):
    weights = obj.weights
    biases = obj.biases
    out = '%s, %s' % (str(weights.shape), str(biases.shape))
    return out


def convert_spatial_average_pooling(obj):
    return ""


def convert_spatial_max_pooling(obj):
    return ""


def convert_unknown(obj):
    return 'UnknownClass'


torch_converters = {}


def convert_obj(typename, obj):
    name_parts = typename.rsplit('.', 1)
    if not name_parts or not name_parts[-1]:
        return
    class_name = name_parts[-1]
    if class_name not in torch_converters:
        return convert_unknown(obj)
    else:
        return torch_converters[class_name](obj)


def add_converter(typename, convert_fn):
    torch_converters[typename] = convert_fn
for mod in [("SpatialAveragePooling", convert_spatial_average_pooling),
            ("SpatialBatchNormalization", convert_spatial_batch_normalization),
            ("SpatialConvolution", convert_spatial_convolution),
            ("SpatialMaxPooling", convert_spatial_max_pooling)]:
    add_converter(mod[0], mod[1])


def process_obj(obj, level=0):
    indent = ''.join(['\t' for s in range(level)])
    if isinstance(obj, torchfile.TorchObject):
        #print(indent + obj.torch_typename())
        print(indent + obj.torch_typename() + ': ' + convert_obj(obj.torch_typename(), obj))
        if obj.modules:
            for x in obj.modules:
                process_obj(x, level+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('torch_file')
    args = parser.parse_args()
    torch_file = args.torch_file

    data = torchfile.load(torch_file, force_8bytes_long=True)

    if data.modules:
        process_obj(data)


if __name__ == '__main__':
    main()
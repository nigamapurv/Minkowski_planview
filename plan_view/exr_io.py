from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Imath
import numpy as np
import OpenEXR

def convert_dtype(imath_pixel_type):
    imath_u32 = Imath.PixelType(Imath.PixelType.UINT)
    imath_f16 = Imath.PixelType(Imath.PixelType.HALF)
    imath_f32 = Imath.PixelType(Imath.PixelType.FLOAT)

    if imath_pixel_type == imath_u32:
        return np.uint32
    elif imath_pixel_type == imath_f16:
        return np.float16
    elif imath_pixel_type == imath_f32:
        return None

def get_shape(input_file):
    exr_image = OpenEXR.InputFile(input_file)
    header = exr_image.header()
    data_window = header["dataWindow"]
    width, height = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    channel_infos = list(header["channels"].items())
    channels = len(channel_infos)

    return height, width, channels

def write_semantic_image(output_file, tensor, output_dtype=Imath.PixelType(Imath.PixelType.HALF)):
    height, width, channels = tensor.shape
    header = OpenEXR.Header(width, height)

    exr_type = Imath.Channel(output_dtype)
    np_type = convert_dtype(output_dtype)

    def conditional_convert(tensor):
        if np_type == tensor.dtype:
            return tensor
        else:
            return tensor.astype(np_type)

    channel_strings = [str(channel) for channel in range(channels)]
    header["channels"] = dict([(channel_string, exr_type) for channel_string in channel_strings])

    data = dict([(channel_strings[channel], conditional_convert(tensor[:, :, channel]).tostring()) for channel in range(channels)])

    exr_file = OpenEXR.OutputFile(output_file, header)
    exr_file.writePixels(data)
    exr_file.close()

def load_semantic_image(input_file, output_dtype=None):
    exr_image = OpenEXR.InputFile(input_file)
    header = exr_image.header()
    data_window = header["dataWindow"]
    width, height = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)

    channel_infos = list(header["channels"].items())
    channel_infos = list(map(lambda x: (int(x[0]), x[1].type), channel_infos))
    channel_infos = sorted(channel_infos, key = lambda x: x[0])
    channel_infos = list(map(lambda x: (str(x[0]), x[1]), channel_infos))
    channel_strings, types = zip(*channel_infos)

    if output_dtype is None:
        output_dtype = convert_dtype(types[0])

    arrays = exr_image.channels(channel_strings)
    np_arrays = []

    for index in range(len(arrays)):
        dtype = convert_dtype(types[index])
        np_array = np.frombuffer(arrays[index], dtype = dtype).reshape(height, width)

        if output_dtype != np_array.dtype:
            np_array = np.cast[output_dtype](np_array)

        np_arrays.append(np_array)

    return np.stack(np_arrays, axis=2)
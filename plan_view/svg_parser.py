from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pupil import io

from typing import List

# Class keys
BACKGROUND = "background"
CORNER = "corner"
DOOR = "door"
WALL = "wall"
ROOM = "room"
FRONT_DOOR = "front_door"
VOID = "void"
OBSTRUCTION = "obstruction"
RESTRICTED_HEIGHT = "restricted_height"
STAIRCASE_GROUND = "staircase_ground"
STORAGE_AREA = "storage_area"
WORKTOP = "worktop"
WINDOW = "window"
ROOM_MEASUREMENT_GUIDE = "room-measurement-guide"
INTERFACE_ADJUSTMENT_AREA = "interface_adjustment_area"

def scale_about_pivot(point, transform_vals):
    effective_scale_x, effective_scale_y, x_pivot, y_pivot = transform_vals
    pivot = np.array([x_pivot, y_pivot, 0], dtype=np.float32).reshape((3, 1))

    point = point - pivot
    point[0] = effective_scale_x * point[0]
    point[1] = effective_scale_y * point[1]
    point = point + pivot

    return point

def rotate_about_pivot(point, transform_vals):
    theta_in_degrees, x_pivot, y_pivot = transform_vals
    pivot = np.array([x_pivot, y_pivot, 0], dtype=np.float32).reshape((3, 1))

    cos_theta = np.cos(np.deg2rad(theta_in_degrees))
    sin_theta = np.sin(np.deg2rad(theta_in_degrees))

    rotation_matrix = np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32
    )

    point = point - pivot
    point = np.dot(rotation_matrix, point)
    point = point + pivot

    return point

def apply_svg_matrix(point, matrix_vals):
    matrix = np.array(
        [
            [matrix_vals[0], matrix_vals[2], matrix_vals[4]],
            [matrix_vals[1], matrix_vals[3], matrix_vals[5]],
            [0, 0, 1]
        ]
    )

    return np.dot(matrix, point)

def convert_points_to_path(points_list):
    path_str = "M "

    for point in points_list[:-1]:
        path_str = path_str + str(int(round(point[0]))) + " "
        path_str = path_str + str(int(round(point[1]))) + " "
        path_str = path_str + "L "

    path_str = path_str + str(int(round(points_list[-1][0]))) + " "
    path_str = path_str + str(int(round(points_list[-1][1]))) + " "
    path_str = path_str + "Z"

    return path_str

def transform_rect_symbol(point, matrix_vals, transform_vals, scale = None):
    offsets = [[0, 0], [0, 100], [100, 100], [100, 0]] # offsets is the size of the un-transformed door symbol
    offsets_npy = [np.array(offset + [0]).reshape((3, 1)) for offset in offsets]

    transformed_corners = []
    for offset_npy in offsets_npy:
        if scale is not None:
            scale_x = np.abs(matrix_vals[0])
            scale_y = np.abs(matrix_vals[3])
            min_scale = np.minimum(scale_x, scale_y)
            if scale_x == min_scale:
                effective_scale_x = 0.4 * scale / scale_x
            else:
                effective_scale_x = 1.0

            if scale_y == min_scale:
                effective_scale_y = 0.4 * scale / scale_y
            else:
                effective_scale_y = 1.0

            offset_npy = scale_about_pivot(offset_npy, [effective_scale_x, effective_scale_y, 50.0, 50.0])

        symbol_corner = point + offset_npy
        symbol_corner = apply_svg_matrix(symbol_corner, matrix_vals)
        symbol_corner = rotate_about_pivot(symbol_corner, transform_vals)
        transformed_corners.append(symbol_corner.reshape((3,))[:2].tolist())

    return convert_points_to_path(transformed_corners)

def parse_d_str(d_str):
    outline = []

    if "M" in d_str and "L" in d_str and "Z" in d_str:
        d_vals = d_str.split(" ")
        num_corners = int(len(d_vals) // 3)
        for idx in range(num_corners):
            outline.append([float(d_vals[3 * idx + 1]), float(d_vals[3 * idx + 2])])

    return outline

def get_name(room_root):
    for title_candidate in room_root.findall("{http://www.w3.org/2000/svg}title"):
        return title_candidate.text

    return "Unknown Room"

def get_corners(room_root):
    outline = []

    found_titles = []
    for title_candidate in room_root.findall("{http://www.w3.org/2000/svg}title"):
        found_titles.append(title_candidate.text.lower())

    is_exterior = any(map(lambda x: ("garden" in x and "winter" not in x) or "exterior" in x, found_titles))

    for path in room_root.findall("{http://www.w3.org/2000/svg}path"):
        d_str = str(path.attrib["d"])

        if path.attrib["class"] == WALL:
            if not is_exterior:
                if "M" in d_str and "L" in d_str and "Z" in d_str:
                    d_vals = d_str.split(" ")
                    num_corners = int(len(d_vals) // 3)
                    for idx in range(num_corners):
                        outline.append([float(d_vals[3 * idx + 1]), float(d_vals[3 * idx + 2])])
                    break

    return outline

def get_measurement_guide(info_root):
    outline = []

    for path in info_root.findall("{http://www.w3.org/2000/svg}path"):
        d_str = str(path.attrib["d"])

        if path.attrib["class"] == ROOM_MEASUREMENT_GUIDE:
            if "M" in d_str and "L" in d_str and "Z" in d_str:
                d_vals = d_str.split(" ")
                num_corners = int(len(d_vals) // 3)
                for idx in range(num_corners):
                    outline.append([float(d_vals[3 * idx + 1]), float(d_vals[3 * idx + 2])])

                break

    return outline

def get_excluded_regions(svg_root):
    symbols = {
        INTERFACE_ADJUSTMENT_AREA: [],
        OBSTRUCTION: []
    }

    # Check if given scene svg ("g") is exterior.
    found_titles = []
    for title_candidate in svg_root.findall("{http://www.w3.org/2000/svg}title"):
        found_titles.append(title_candidate.text.lower())

    for path in svg_root.findall("{http://www.w3.org/2000/svg}path"):
        d_str = str(path.attrib["d"])

        if path.attrib["class"] == OBSTRUCTION:
            symbols[OBSTRUCTION].append(parse_d_str(d_str))
        elif path.attrib["class"] == INTERFACE_ADJUSTMENT_AREA:
            symbols[INTERFACE_ADJUSTMENT_AREA].append(parse_d_str(d_str))

    return symbols
  
def per_scene_iterator(svg_root):
    for candidate in svg_root.findall("{http://www.w3.org/2000/svg}g"):
        if "room" in candidate.attrib["class"]:
            if "id" in candidate.keys() and candidate.attrib["id"]:
                scene_id = str(candidate.attrib["id"])
                yield candidate, scene_id
            else :
                yield None

def translate_polygon(polygon, translation):
    dx, dy = translation
    return [[x + dx, y + dy] for x, y in polygon]

def normalize_polygon(polygon, normalization_scale=100.0):
    return [[x / normalization_scale, y / normalization_scale] for x, y in polygon]

def get_limits(view_box):
    lower_limit = np.array(view_box[0:2]) / 100.0
    upper_limit = lower_limit + np.array(view_box[2:4]) / 100.0

    return lower_limit, upper_limit

def get_inverse_projection_matrix(view_box):
    lower_limit = np.array(view_box[0:2]) / 100.0
    width = int(np.ceil(view_box[2]))
    height = int(np.ceil(view_box[3]))

    shape = np.array([float(width), float(height)])

    focal_length = 100.0 / shape
    principal_point = -lower_limit * 100.0 / shape

    projection_matrix = np.array(
        [
            [focal_length[0], 0.0, 0.0, principal_point[0]],
            [0.0, focal_length[1], 0.0, principal_point[1]],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    )

    inverse_projection_matrix = np.array(
        [
            [1.0 / focal_length[0], 0.0, - principal_point[0] / focal_length[0]],
            [0.0, 1.0 / focal_length[1], - principal_point[1] / focal_length[1]],
            [0.0, 0.0, 1.0]
        ]
    )

    return width, height, inverse_projection_matrix, projection_matrix

def parse_ground_truth_outlines(capture: dict):
    for section in capture["sections"]:
        svg_ref = section["groundTruthFloorplan"]["mediaReference"]

        print("Downloading svg from s3://{}/{}".format(svg_ref["bucket"], svg_ref["path"]))
        tree = io.load_svg_from_s3(svg_ref["bucket"], svg_ref["path"])
        root = tree.getroot()

        view_box = [float(value) for value in root.attrib["viewBox"].split(" ")]
        lower_limit = np.array(self.view_box[0:2]) / 100.0
        width = int(np.ceil(self.view_box[2]))
        height = int(np.ceil(self.view_box[3]))
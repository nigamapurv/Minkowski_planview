from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import open3d
import pupil_vision
import MinkowskiEngine as ME
import numpy as np
import os
import scipy.sparse
import matplotlib.pyplot as plt

from plan_view import exr_io, svg_parser
from pupil import io

from plan_view.get_colormaps import create_generic_colormap
import torch

import cv2

# 878427a4d33d4a429852b59b3834ae03

class Asset:
    def __init__(self, bucket, path, filename):
        self.bucket = bucket
        self.path = path
        self.filename = filename

def _make_key(capture_id, item_id):
    return "{}+{}".format(capture_id, item_id)

def find_task_in_workflow_context(workflow_context, name):
    for task in workflow_context["tasks"]:
        if task["name"] == name:
            return task

    raise Exception("Task {} not found in workflow context".format(name))

def scene_iterator(response):
    for capture in response:
        capture_id = capture["captureId"]

        for scene in capture["scenes"]:
            scene_id = scene["id"]

            yield capture_id, scene_id, scene

def section_iterator(response):
    for capture in response:
        capture_id = capture["captureId"]

        for section in capture["sections"]:
            section_id = section["id"]

            yield capture_id, section_id, section

def get_point_cloud_reference_and_transformation(scan_data, key_priority=["pointCloudSubsampledTiltedWithNormals", "pointCloudTilted", "pointCloud"]):
    if "registrationResult" in scan_data:
        registration_result = scan_data["registrationResult"]

        if "transformation" in registration_result:
            transformation = registration_result["transformation"]

            for key in key_priority:
                if key in scan_data:
                    reference = scan_data[key]

                    if reference is not None and "bucket" in reference and "path" in reference:
                        return reference, transformation

    return None

class Example:
    def __init__(self, scans, transformations, floor_plan, label, capture_id, merged_pointCloud):
        self.scans = scans
        self.transformations = transformations
        self.floor_plan = floor_plan
        self.label = label
        self.capture_id = capture_id
        self.merged_pointloud = merged_pointCloud

    def asset_iterator(self):
        for scan in self.scans:
            yield scan

        yield self.floor_plan
        yield self.label

class DataLoader:
    def __init__(self, execution_id, work_dir, num_shards=1):
        self.execution_id = execution_id
        self.work_dir = work_dir
        self.num_shards = num_shards

        self.workflow_context = io.load_json_from_s3("darkroom-plan-view-datasets", f"executions/{self.execution_id}/workflow-context.json")
        self.response = self.parse_data()
        self.examples = self.make_examples()
        # self.savePointClouds()

        # self.download_assets()


    def parse_data(self):
        print("Parsing Response ")
        registration_task = find_task_in_workflow_context(self.workflow_context, "registration")
        plan_view_label_generation_task = find_task_in_workflow_context(self.workflow_context, "plan-view-label-generation")

        response = []

        for registration_response_reference, plan_view_label_response_reference in zip(registration_task["responses"][:self.num_shards], plan_view_label_generation_task["responses"][:self.num_shards]):
            registration_response = io.load_json_from_s3(**registration_response_reference)
            plan_view_label_response = io.load_json_from_s3(**plan_view_label_response_reference)

            _local_cache = {}

            for capture_id, item_id, item in scene_iterator(registration_response):
                key = _make_key(capture_id, item_id)
                _local_cache[key] = item

            for capture_id, item_id, item in scene_iterator(plan_view_label_response):
                key = _make_key(capture_id, item_id)
                if key not in _local_cache:
                    raise Exception("Registration response missing for key {}".format(key))
                else:
                    other_item = _local_cache[key]
                    if "planViewLabel" in item:
                        other_item["planViewLabel"] = item["planViewLabel"]

            for capture_id, item_id, section in section_iterator(plan_view_label_response):
                ground_truth_floorplan = section["groundTruthFloorplan"]

                for scene in section["scenes"]:
                    key = _make_key(capture_id, scene["id"])
                    if key in _local_cache:
                        item = _local_cache[key]
                        item["groundTruthFloorplan"] = ground_truth_floorplan

            response.extend(registration_response)

        return response

    def make_examples(self):
        examples = []

        for capture_id, item_id, item in scene_iterator(self.response):
            print("Making Example for cap_id {}".format(capture_id))
            scans = []
            transformations = []

            for scan in item["scans"]:
                scan_reference_and_transformation = get_point_cloud_reference_and_transformation(scan)
                if scan_reference_and_transformation is not None:
                    scan_reference, scan_transformation = scan_reference_and_transformation
                    scan_bucket = scan_reference["bucket"]
                    scan_path = scan_reference["path"]
                    scan_filename = os.path.join(self.work_dir, scan_path)
                    transformation = np.array(scan_transformation).reshape(4, 4).transpose()

                    scans.append(Asset(scan_bucket, scan_path, scan_filename))
                    transformations.append(transformation)

            if ('groundTruthFloorplan' not in item):
                continue
            floor_plan_bucket = item["groundTruthFloorplan"]["mediaReference"]["bucket"]
            floor_plan_path = item["groundTruthFloorplan"]["mediaReference"]["path"]
            floor_plan_filename = os.path.join(self.work_dir, floor_plan_path)
            floor_plan = Asset(floor_plan_bucket, floor_plan_path, floor_plan_filename)

            if('planViewLabel' not in item):
                continue
            label_bucket = item["planViewLabel"]["bucket"]
            label_path = item["planViewLabel"]["path"]
            label_filename = os.path.join(self.work_dir, label_path)
            label = Asset(label_bucket, label_path, label_filename)

            scene_name = os.path.basename(label_filename).split('_')[1].split('.')[0]
            merged_pointCloud = os.path.join(self.work_dir, capture_id, '{}.npy'.format(scene_name))


            example = Example(scans, transformations, floor_plan, label, capture_id, merged_pointCloud)
            examples.append(example)

        return examples

    def download_assets(self):
        for example in self.examples:
            io.download_assets(example.asset_iterator)

    def savePointClouds(self):


        for example in self.examples:
            points = []
            intensities = []
            normals = []
            scan_origins = []

            example: Example = example
            scene_name = os.path.basename(example.label.filename).split('_')[1].split('.')[0]

            if os.path.exists( os.path.join(self.work_dir, example.capture_id, '{}.npy'.format(scene_name)) ):
                print("Merged Point Cloud Already exists, skiping ")
                continue
            else:
                if(not os.path.exists(os.path.join(self.work_dir, example.capture_id))):
                    os.mkdir(os.path.join(self.work_dir, example.capture_id))

                print("Merging Point Cloud for {}".format(os.path.join(self.work_dir, example.capture_id, '{}.npy'.format(scene_name)) ))

                floor_plan = io.load_svg_from_file(example.floor_plan.filename)
                #
                view_box = [float(value) for value in floor_plan.getroot().attrib["viewBox"].split(" ")]
                width = int(np.ceil(view_box[2]))
                height = int(np.ceil(view_box[3]))

                if width * height > 2000 * 1500:
                    print("Skipping Florplan, to large ....{} {}".format(width,height))
                    continue

                filenames = [scan.filename for scan in example.scans]
                transformations = example.transformations

                if len(filenames) == 0:
                    continue

                for filename, transformation in zip(filenames, transformations):
                    point_cloud = pupil_vision.read_point_cloud(filename)
                    point_cloud.estimate_normals()
                    point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
                    point_cloud.transform(transformation)

                    points.append(np.asarray(point_cloud.points))
                    intensities.append(np.asarray(point_cloud.intensities))
                    normals.append(np.asarray(point_cloud.normals))

                    scan_origins.append(transformation[0:3, 3])

                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(np.concatenate(points, axis=0))

                main_component_mask = filter_by_connected_components(point_cloud, scan_origins)

                coords = np.concatenate(points, axis=0)[main_component_mask, ...]
                np_points = coords
                np_intensities = np.concatenate(intensities, axis=0)[main_component_mask, np.newaxis]
                np_normals = np.concatenate(normals, axis=0)[main_component_mask]
                np_combined_ptCl = np.concatenate([np_points, np_intensities, np_normals], axis=1).astype(np.float16)
                print("Saving Merged Point Cloud {} ".format(os.path.join(self.work_dir,example.capture_id, '{}.npy'.format(scene_name))))
                np.save(os.path.join(self.work_dir,example.capture_id, '{}.npy'.format(scene_name)), np_combined_ptCl)

        print("Done cnversion of pointclouds to Numpy format")

    def iterator(self):
        voxel_size = 0.08

        for example in itertools.chain(self.examples, self.examples, self.examples, self.examples, self.examples):
            example: Example = example
            # print(example.label.path)
            # if (example.label.path != "plan-view-label-generation/5c6273962ffa2b9fa59ee16006c4b5a92dae5e16/2514--3c770956-c09d-4ef2-9fac-2ddefc56d508_e9d13996-59c0-48fe-a396-e51740fcd218.exr"):
            #     continue
            label = exr_io.load_semantic_image(example.label.filename)
            floor_plan = io.load_svg_from_file(example.floor_plan.filename)

            canvas = np.zeros(label.shape[0:2], dtype=np.uint8)
            colour_map = create_generic_colormap(label.shape[2]).astype(np.uint8)
            for class_index in range(label.shape[2]):
                channel = label[:, :, class_index]
                channel[channel==class_index] = 255
                # kernel = np.ones((35, 35), np.uint8)
                # channel = cv2.dilate(np.asarray(channel, dtype='uint8'), kernel, iterations=1)
                canvas[channel > 0] = class_index

            # plt.subplot(121)
            # plt.imshow(colour_map[canvas])

            width, height, inverse_projection_matrix, projection_matrix = svg_parser.get_inverse_projection_matrix([float(value) for value in floor_plan.getroot().attrib["viewBox"].split(" ")])
            if width * height > 2000 * 1500:
                continue

            filenames = [scan.filename for scan in example.scans]
            transformations = example.transformations
            scene_name = os.path.basename(example.label.filename).split('_')[1].split('.')[0]
            # savePointClouds(filenames, transformations, scene_name, self.work_dir)
            if len(filenames) == 0:
                continue
            #
            input_tensor, ptCld_labels,  min_coords, max_coords = generate_input_sparse_tensor(filenames, transformations, voxel_size, projection_matrix, canvas,width, height, example.merged_pointloud)
            # diff_coords = max_coords - min_coords
            # dx = diff_coords[0] / voxel_size
            # dy = diff_coords[1] / voxel_size
            # minkowski_projection_matrix = np.array(
            #     [
            #         [1.0 / (voxel_size * dx), 0.0, -min_coords[0] / (voxel_size * dx)],
            #         [0.0, 1.0 / (voxel_size * dy), -min_coords[1] / (voxel_size * dy)],
            #         [0.0, 0.0, 1.0]
            #     ]
            # )
            # u, v = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
            # uvw_label = np.stack([u, v, np.ones_like(u)], axis=2)
            # xyz = np.matmul(uvw_label, inverse_projection_matrix.transpose())
            # uv_minkowski = np.expand_dims(2.0 * np.matmul(xyz, minkowski_projection_matrix.transpose())[:, :, 0:2] - 1.0, axis=0)
            # uv_minkowski = np.stack([uv_minkowski[..., 1], uv_minkowski[..., 0]] , axis=3)
            yield input_tensor, ptCld_labels

def safe_div(x, y):
    return x / np.where(y > 0, y, np.ones_like(y))

def filter_by_connected_components(point_cloud, scan_origins, radius=0.1, max_nn=10, max_num_nearby_points=300, always_include_radius=7.5):
    kdtree = open3d.geometry.KDTreeFlann(point_cloud)

    points = np.asarray(point_cloud.points)
    num_points, _ = points.shape

    # print("Filtering point cloud of shape {} using connected components".format(points.shape))
    # print("Using parameters (radius = {}, max_nn = {}, max_num_nearby_points = {}, always_include_radius = {})".format(radius, max_nn, max_num_nearby_points, always_include_radius))

    # Convert origins to a 1 x 3 x D array
    origins = np.expand_dims(np.transpose(np.array(scan_origins), axes=[1, 0]), axis=0)

    # Convert points to a N x 3 x 1 array
    tiled_points = np.expand_dims(points, axis=2)

    max_batch_size = 100000
    num_batches = int(np.ceil(float(num_points) / float(max_batch_size)))
    min_distances = np.zeros((num_points,), dtype = np.float32)
    for batch_index in range(num_batches):
        batch_start = batch_index * max_batch_size
        batch_end = batch_start + min(max_batch_size, num_points - batch_start)
        distances = np.linalg.norm(origins - tiled_points[batch_start:batch_end, ...], axis=1, ord=2)
        min_distances[batch_start:batch_end] = np.min(distances, axis=1)

    sorted_point_indices = np.argsort(min_distances)

    always_included_indices = np.where(min_distances < always_include_radius)

    # Find neighbours of points
    filtered_triplets = pupil_vision.find_triplets(kdtree, point_cloud, radius, max_nn) # [T, 3]

    # Find closest points to origin
    nearby_point_triplet_list = []
    num_nearby_points = min(max_num_nearby_points, sorted_point_indices.shape[0])
    for index in range(num_nearby_points):
        for pair_index in range(index + 1, num_nearby_points):
            if sorted_point_indices[index] > sorted_point_indices[pair_index]:
                nearby_point_triplet_list.append([sorted_point_indices[index], sorted_point_indices[pair_index], 1])
            else:
                nearby_point_triplet_list.append([sorted_point_indices[pair_index], sorted_point_indices[index], 1])

    nearby_point_triplets = np.array(nearby_point_triplet_list, dtype = np.int32)

    # Concatenate triplets from NN search and closest points search
    all_triplets = np.concatenate([filtered_triplets, nearby_point_triplets], axis=0)
    del filtered_triplets
    del nearby_point_triplets
    del nearby_point_triplet_list

    # print("Constructing sparse matrix with (shape = {}, nnz = {})".format((num_points, num_points), all_triplets.shape[0]))
    graph = scipy.sparse.coo_matrix((all_triplets[:, 2], (all_triplets[:, 0], all_triplets[:, 1])), shape=(num_points, num_points))

    # print("Finding connected components from sparse matrix")
    num_components, components = scipy.sparse.csgraph.connected_components(graph, directed = False)

    # print("Found {} components".format(num_components))
    always_included_components = np.unique(components[always_included_indices])
    main_component_mask = np.isin(components, always_included_components)

    return main_component_mask

def min_distances_from_scan_locations(points, scan_origins):
    num_points, _ = points.shape

    # Convert origins to a 1 x 3 x D array
    origins = np.expand_dims(np.transpose(np.array(scan_origins), axes=[1, 0]), axis=0)

    # Convert points to a N x 3 x 1 array
    tiled_points = np.expand_dims(points, axis=2)

    max_batch_size = 100000
    num_batches = int(np.ceil(float(num_points) / float(max_batch_size)))
    min_distances = np.zeros((num_points,), dtype = np.float32)
    for batch_index in range(num_batches):
        batch_start = batch_index * max_batch_size
        batch_end = batch_start + min(max_batch_size, num_points - batch_start)
        distances = np.linalg.norm(origins - tiled_points[batch_start:batch_end, ...], axis=1, ord=2)
        min_distances[batch_start:batch_end] = np.min(distances, axis=1)

    return min_distances

def make_homogeneous(points):
    num_points, _ = points.shape
    return np.concatenate([points, np.ones(shape = [num_points, 1], dtype = points.dtype)], axis = 1)


def load_files(filenames, transformations, voxel_size, projection_matrix, canvas,width, height, merged_pointCloud):
    points = []
    intensities = []
    normals = []
    scan_origins = []

    # for filename, transformation in zip(filenames, transformations):
    #      point_cloud = pupil_vision.read_point_cloud(filename)
    #      point_cloud.estimate_normals()
    #      point_cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    #      point_cloud.transform(transformation)
    #
    #      points.append(np.asarray(point_cloud.points))
    #      intensities.append(np.asarray(point_cloud.intensities))
    #      normals.append(np.asarray(point_cloud.normals))
    #
    #      scan_origins.append(transformation[0:3, 3])

    for transformation in transformations:

         scan_origins.append(transformation[0:3, 3])


    # point_cloud = open3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(np.concatenate(points, axis=0))



    #main_component_mask = filter_by_connected_components(point_cloud, scan_origins)

    # print(f"Main component has {np.count_nonzero(main_component_mask)} points")
    np_point_cloud = np.load(merged_pointCloud)
    coords = np_point_cloud[:,0:3]

    print(np.shape(coords), np.shape(intensities), np.shape(normals), np.shape(scan_origins))


    # coords = np.concatenate(points, axis=0)[main_component_mask, ...]
    # point_cloud_filered = open3d.geometry.PointCloud()
    # point_cloud_filered.points = open3d.utility.Vector3dVector(coords)
    # np_point_cloud = np.asarray(point_cloud_filered.points)


    homogeneous_point_cloud = make_homogeneous(coords)
    projected_point_cloud = np.matmul(homogeneous_point_cloud, projection_matrix.transpose())
    uv = projected_point_cloud[:, 0:2]
    uv = np.asarray(np.matmul(uv, np.asarray(((width, 0), (0, height)))), dtype='uint32')

    valid_uv_indices = np.intersect1d(np.where(uv[:,0]<width) , np.where( uv[:,1]<height))
    uv = uv[valid_uv_indices]
    coords = coords[valid_uv_indices]
    colors = np.zeros(np.shape(coords))


    labels_flat = np.reshape(canvas, [width*height, 1])
    i = uv[:,0]+uv[:,1]*width
    ptCld_labels = np.take(labels_flat,i)

    bg_ind = np.where(ptCld_labels == 0)
    bg = uv[bg_ind[0]]
    colors[bg_ind] = [255, 255, 0]


    room_ind = np.where(ptCld_labels == 1)
    room = uv[room_ind[0]]
    colors[room_ind] = [0, 0, 255]


    wall_ind = np.where(ptCld_labels == 2)
    walls = uv[wall_ind[0]]
    colors[wall_ind] = [0, 255, 0]

    corners_ind = np.where(ptCld_labels == 3)
    corners = uv[corners_ind[0]]
    colors[corners_ind] = [255, 0, 0]



    # plt.subplot(122)
    # plt.scatter(corners[:,0], corners[:,1], s=1,color="red")
    # plt.scatter(walls[:,0], walls[:,1], s=1, color="green")
    # plt.scatter(room[:,0], room[:,1], s=1, color="blue")
    # plt.gca().set_xlim(xmin=0)
    # plt.gca().set_ylim(ymin=0)
    # plt.gca().invert_yaxis()
    # plt.axis('equal')
    # plt.show()
    #
    # point_cloud_cliped = open3d.geometry.PointCloud()
    # point_cloud_cliped.points = open3d.utility.Vector3dVector(coords)
    # point_cloud_cliped.colors =  open3d.utility.Vector3dVector(colors)
    # open3d.visualization.draw_geometries([point_cloud_cliped])

    #intensities = np.expand_dims(2.0 * np.concatenate(intensities, axis=0) - 1.0, axis=1)[main_component_mask, ...][valid_uv_indices]
    #normals = np.concatenate(normals, axis=0)[main_component_mask, ...][valid_uv_indices]
    intensities = np.expand_dims(2.0 * np_point_cloud[:, 3] - 1, axis=1)[valid_uv_indices]
    normals = np_point_cloud[:, 4:7][valid_uv_indices]
    min_distances = np.expand_dims(min_distances_from_scan_locations(coords, scan_origins), axis=1)

    features = np.concatenate([intensities, normals, min_distances], axis=1)

    quantized_coords = np.floor(coords / voxel_size)
    indices = ME.utils.sparse_quantize(quantized_coords)

    return quantized_coords[indices], features[indices], ptCld_labels[indices], np.min(coords, axis=0), np.max(coords, axis=0)

def generate_input_sparse_tensor(filenames, transformations, voxel_size, projection_matrix, canvas,width, height, merged_pointCloud):
    # Create a batch, this process is done in a data loader during training in parallel.
    quantized_coords, corresponding_features, ptCld_labels,  min_coords, max_coords = load_files(filenames, transformations, voxel_size, projection_matrix, canvas,width, height, merged_pointCloud)
    batch = [(quantized_coords, corresponding_features)]
    coordinates_, features_ = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, features_)

    # Normalize features and create a sparse tensor
    return ME.SparseTensor(features, coords=coordinates), torch.from_numpy(np.array(ptCld_labels)).long(),  min_coords, max_coords
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import open3d
import pupil_vision

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import matplotlib.pyplot as plt
import open3d

import time

from examples.resnet import ResNetBase

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from torch.optim import Adam, SGD

from plan_view.dataloader import DataLoader

from scipy.special import expit

class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            has_bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat((out, out_b3p8))
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat((out, out_b2p4))
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat((out, out_b1p2))
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat((out, out_p1))
        out = self.block8(out)

        return self.final(out)


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

VALID_CLASS_IDS = [
    0,1, 2, 3
]
PUPIL_COLOR_MAP = {
    0: (255., 255., 0.),
    1: (0., 0., 255.),
    2: (0., 255., 0.),
    3: (255., 0., 0.),

}

if __name__ == '__main__':
    work_dir = "/media/apurvnigam/Storage/tmp/MinkowskiEngine"
    dataloader = DataLoader("9ba730a34e594841a84aeac52eed12d4", work_dir, num_shards=1)
    criterion = nn.CrossEntropyLoss()
    net = MinkUNet34C(in_channels=5, out_channels=4, D=3)

    model_dict = torch.load("network_iter2090.pth")
    net.load_state_dict(model_dict)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    initial_index = 2090




    for index, (input_tensor, label) in enumerate(dataloader.iterator()):
        index=index+initial_index
        # torch.cuda.empty_cache()
        # optimizer.zero_grad()
        #
        # try:
        #
        #     input = input_tensor.to(device)
        #     label = label.to(device)
        #
        #     # Forward
        #     output = net(input)
        #
        #
        #     # if index % 1 == 0:
        #     #
        #     #     _, pred = output.F.max(1)
        #     #     pred = pred.cpu().numpy()
        #     #     colors = np.array([PUPIL_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
        #     #     pred_pcd = open3d.geometry.PointCloud()
        #     #     coordinates = output.C.numpy()[:, :3]  # last column is the batch index
        #     #     pred_pcd.points = open3d.utility.Vector3dVector(coordinates * 0.02)
        #     #     pred_pcd.colors = open3d.utility.Vector3dVector(colors / 255)
        #     #     # vis = open3d.visualization.Visualizer()
        #     #     # vis.create_window()
        #     #     open3d.visualization.draw_geometries([pred_pcd])
        #
        #         #
        #         # vis.add_geometry(pred_pcd)
        #         # vis.update_geometry()
        #         # vis.poll_events()
        #         # vis.update_renderer()
        #         #
        #         # time.sleepp(2)
        #         # vis.destroy_window()
        #     if index%300 ==0:
        #         torch.save(net.state_dict(), 'network_iter{}.pth'.format(index))
        #
        #
        #         # pred_pcd.points = open3d.Vector3dVector([])
        #         # pred_pcd.colors = open3d.Vector3dVector([])
        #         # vis.update_geometry()
        #         # vis.poll_events()
        #         # vis.update_renderer()
        #
        #
        #
        #
        #     # Loss
        #     loss = criterion(output.F, label)
        #     print('Iteration: ', index, ', Loss: ', loss.item())
        #
        #     # Gradient
        #     loss.backward()
        #     optimizer.step()
        #
        # except Exception as e:
        #     print("Exception captured")
        #     import traceback
        #     traceback.print_exc(e)
        #     pass

        # Saving and loading a network
    # torch.save(net.state_dict(), 'test.pth')
    # net.load_state_dict(torch.load('test.pth'))


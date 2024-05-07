#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path       #将传入的 args 对象中的 model_path 属性赋值给 self.model_path，表示模型的路径。
        self.loaded_iter = None                 #初始化 self.loaded_iter 为 None，用于存储已加载的模型的迭代次数。
        self.gaussians = gaussians              #将传入的高斯模型对象赋值给 self.gaussians 属性。
        
        if load_iteration:                      #可选参数，默认为 None。如果提供了值，它将被用作已加载模型的迭代次数。
            if load_iteration == -1:            #如果没有提供 load_iteration，则将点云数据和相机信息保存到文件中。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))     #输出加载模型的迭代次数的信息。

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}     #存储视频序列相机信息
        
        # 根据场景的类型（Colmap 或 Blender）加载相应的场景信息，并划分训练集测试集，存储在 scene_info 变量中。场景存储的信息包括初始化点云，训练的相机信息，测试的相机信息，生成的video相机信息，(中心相机信息和相机分布的边界长度),点云路径,视频的最长时间
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime   #设置视频的最大时间
        self.dataset_type = dataset_type    #设置数据集类型
        self.cameras_extent = scene_info.nerf_normalization["radius"]      #训练照片中的相机分布的对角线     
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)    #初始化训练相机的信息
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        
        #是否需要添加点云
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)   #变形场网络的初始设置
        
        # 保存点云数据和相机信息：
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)   #初始化每个点的Gaussian，根据初始化的点云来做

    def save(self, iteration, stage):   #该方法用于保存点云数据到文件，其参数 iteration 指定了迭代次数。
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):       #返回训练相机的列表，可以根据指定的缩放因子 scale 获取相应分辨率的相机列表。
        return self.train_camera

    def getTestCameras(self, scale=1.0):        #返回测试相机的列表，可以根据指定的缩放因子 scale 获取相应分辨率的相机列表。
        return self.test_camera
    def getVideoCameras(self, scale=1.0):       
        return self.video_camera
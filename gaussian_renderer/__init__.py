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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

# 这段代码是一个用于渲染场景的函数，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()    #尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度。
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)#创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上。

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)#获取预计算的三维协方差矩阵。
    else: #获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:   #如果是第一阶段则直接跳过deformation
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:   #如果是第二阶段，则需要过一个deformation网络，加上网络预测的
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换。
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) #将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))#计算相机中心到每个点的方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)#计算相机中心到每个点的方向向量归一化。
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)#使用SH特征将方向向量转换为RGB颜色。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) #将RGB颜色的范围限制在0到1之间。
        else:
            pass
            # shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径。
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}


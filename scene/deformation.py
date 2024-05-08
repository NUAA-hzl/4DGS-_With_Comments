import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)      #这里的HexPlane有点神奇
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim       #这里似乎是属性的维度
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]      #64,64
        
        for i in range(self.D-1):       #根据设定的深度创建mlp
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))         #关于位置的deform:1.Relu;2.Linear(64,64);3.Relu;4.Linear(64,3)-->xyz
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))      #关于缩放比例的deform:1.Relu;2.Linear(64,64);3.Relu;4.Linear(64,3)--->xyz(scale)
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))   #关于旋转矩阵的deform:1.Relu;2.Linear(64,64);3.Relu;4.Linear(64,4)--->旋转的四元数
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))     #关于透明度的deform:1.Relu;2.Linear(64,64);3.Relu;4.Linear(64,1)--->opacity
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))      #关于球谐函数的deform:1.Relu;2.Linear(64,64);3.Relu;4.Linear(64,16*3)--->这里我不太懂为什么是16*3

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):      #看起来是将时间信息整合到空间信息中

        if self.no_grid:        #如果使用网格
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:                   #如果不使用网格

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])    #前三位的xyz是真实xyz，time_sel这里也是真实time，这里送进HexPlane似乎是在按照HexPlane中的操作，将xyzt合成的4D张量分解成多个低维的张量最后再融合出来，这里出来的shape=[pointNum,64]
            # breakpoint()
            if self.grid_pe > 1:        #这里还可以将出来的融合了时间和空间信息的feature进行一个带位置信息的embedding
                grid_feature = poc_fre(grid_feature,self.grid_pe)       
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   #到这里是经过了HexPlane出来的融合了时空信息的feature，然后过了一层linear的Mlp，维度为64-->64
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:    #如果没有输入时间，代表这个是静态场景
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)      #这里的time_ferture我没懂是什么变量

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)   #这里只用到了xyz、旋转、缩放、时间信息,但是实际上只是经过了一个HexPlane，将xyz信息与时间信息很好的融合在了一起，然后又过了一层MLP线性层出来了
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)  #这里我没看懂有什么用
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)        #将经过了HexPlane和Mlp的feature送到x的deformation里面去，过……(详见pos_deform)中的定义，出来xyz
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx  #实现一个xyz+Delta(xyz)的作用 
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)     #和上面一样

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)  #和上面一样

            rotations = torch.zeros_like(rotations_emb[:,:4])   
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) #和上面一样
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])#和上面一样

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs     #这里返回了经过了deformation之后的xyz,s,r,o,shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width                      #网络的宽度，一般就是进去之后第一层linear的输出尺寸--64
        timebase_pe = args.timebase_pe                  #4
        defor_depth= args.defor_depth                   #0
        posbase_pe= args.posebase_pe                    #10
        scale_rotation_pe = args.scale_rotation_pe      #2
        opacity_pe = args.opacity_pe                    #2
        timenet_width = args.timenet_width              #64
        timenet_output = args.timenet_output            #时间net的输出维度---32
        grid_pe = args.grid_pe                          #0
        times_ch = 2*timebase_pe+1                      #时间的通道9
        self.timenet = nn.Sequential(                   #时间网络的结构:1.全连接层(输入为9，输出为64);2.Relu激活层;3.全连接层(输入为64，输出为32)
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))                                       #下面input_ch的输入维度构建我没有看懂
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))       #1,2,4,8
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))         #1,2,4,8,16……512
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))     #1,2
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))     #1,2 猜测这里是用来保存各个信息的位置编码的
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)     #shs.shape=[point,16,3]，这里的16我一直没懂是什么
        point_emb = poc_fre(point,self.pos_poc)             #这里的pos_poc是结合了位置信息的embedding的操作吧，出来的维度是结合了位置信息的embedding，shape为[pointNum,原本的维度+poc_buf*原本维度*2]，这里是[pointNum,3+3*10*2]
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)  #缩放矩阵为3元，所以出来是[pointNum,3+2*3*2]
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc) #旋转矩阵是四元数，poc_buf维度为2，所以shape=[pointNum,4+4*2*2]
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,     #这个网络输入的是经过embedding之后的xyz[pointNum,63]，缩放矩阵[pointNum.15]，旋转矩阵[pointNum,20]，透明度[pointNum,1]，shs系数[pointNum,16,3],time_sel[pointNum,1]
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):                    #这个函数我没有搞懂是什么意思，感觉像是结合了位置信息的embedding,将原本的xyz同对应的位置信息分别扩展，然后用sin和cos进行一个计算，得到对应的poc_buf*原本信息维度(例如在xyz中是3)，最终拼接原本的信息和过了sin和cos的信息，出来一个结合了位置信息的embedding，shape为[pointNum,原本的维度+poc_buf*原本维度*2]

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)       #在point的xyz方面，将[point,3]-->[point,3,1]*[10]--->[point,3,10]-->[point,30]
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
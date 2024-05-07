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

from argparse import ArgumentParser, Namespace
import sys
import os

#一个空的父类，用于存储参数组
class GroupParams:
    pass

#一个基类，
# 用于创建命令行参数组，
# 并提供了一个方法来从解析后的命令行参数中提取特定的参数组。
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

#用于存储模型加载参数的类
class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3  #球谐函数的阶数
        self._source_path = ""  #数据源的路径
        self._model_path = ""   #模型的路径
        self._images = "images" #图片文件夹的名称
        self._resolution = -1   #图片的分辨率
        self._white_background = True   #图片是否使用白色背景
        self.data_device = "cuda"   #数据加载到的设备
        self.eval = True    #是否启用评估模式，这里初始值为什么是True
        self.render_process=False   #是否启用渲染进程
        self.add_points=False       #是否添加点
        self.extension=".png"       #图片的拓展名
        self.llffhold=8             #在LLFF数据加载中保留从帧数
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


#用于存储数据加载流程参数的类
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False     #是否使用python转换球谐函数
        self.compute_cov3D_python = False   #是否使用python计算3D协方差
        self.debug = False  #是否开启调试模型
        super().__init__(parser, "Pipeline Parameters")
        

#用于存储模型隐藏层参数的类
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64     #网络的宽度
        self.timebase_pe = 4    #时间基底的元素数量
        self.defor_depth = 1    #变形网络的深度
        self.posebase_pe = 10   #姿态基地的元素数量
        self.scale_rotation_pe = 2  #缩放和旋转基地的元素数量
        self.opacity_pe = 2     #透明度基地的元素数量
        self.timenet_width = 64     #时间网络的宽度
        self.timenet_output = 32    #时间网络的输出大小
        self.bounds = 1.6           #边界值
        self.plane_tv_weight = 0.0001   #平面总变差权重
        self.time_smoothness_weight = 0.01  #时间平滑度权重
        self.l1_time_planes = 0.0001    #L1时间平面的权重
        self.kplanes_config = {         #K-Planes配置字典
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]        #多分辨率设置
        self.no_dx=False    #是否不使用梯度
        self.no_grid=False  #是否不使用网络
        self.no_ds=False    #是否不适用采用
        self.no_dr=False    #是否不适用旋转
        self.no_do=True     #是否不使用透明度
        self.no_dshs=True   #是否不使用球谐函数
        self.empty_voxel=False  #是否使用空体素
        self.grid_pe=0      #网格基地的元素数量
        self.static_mlp=False   #是否使用静态MLP
        self.apply_rotation=False   #是否应用旋转

        
        super().__init__(parser, "ModelHiddenParams")
        
        
#存储优化参数的类
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False       #是否使用数据加载器
        self.zerostamp_init=False   #是否使用时间戳初始化
        self.custom_sampler=None    #自定义的采样器
        self.iterations = 30_000    #总迭代次数
        self.coarse_iterations = 3000   #粗迭代次数
        self.position_lr_init = 0.00016 #位置学习率初始值
        self.position_lr_final = 0.0000016  #位置学习率最终值
        self.position_lr_delay_mult = 0.01  #位置学习率延迟乘数
        self.position_lr_max_steps = 20_000 #位置学习率最大步数
        self.deformation_lr_init = 0.00016  #变形学习率初始值
        self.deformation_lr_final = 0.000016    #变形学习率最终值
        self.deformation_lr_delay_mult = 0.01   #变形学习率延迟乘数
        self.grid_lr_init = 0.0016          #网格学习率初始值
        self.grid_lr_final = 0.00016        #网格学习率最终值

        self.feature_lr = 0.0025            #特征学习率
        self.opacity_lr = 0.05              #透明度学习率
        self.scaling_lr = 0.005             #缩放学习率
        self.rotation_lr = 0.001            #旋转学习率
        self.percent_dense = 0.01           #密集百分比
        self.lambda_dssim = 0               #DSSIM权重
        self.lambda_lpips = 0               #LPIPS权重
        self.weight_constraint_init= 1          #权重约束初始值
        self.weight_constraint_after = 0.2      #权重约束之后值
        self.weight_decay_iteration = 5000      #权重衰减迭代次数
        self.opacity_reset_interval = 3000      #透明度重置间隔
        self.densification_interval = 100       #密集化间隔
        self.densify_from_iter = 500            #从迭代开始密集化
        self.densify_until_iter = 15_000        #直到迭代密集化
        self.densify_grad_threshold_coarse = 0.0002     #粗密集化梯度阈值
        self.densify_grad_threshold_fine_init = 0.0002      #细密集化初始梯度阈值
        self.densify_grad_threshold_after = 0.0002      #细密集化之后梯度阈值
        self.pruning_from_iter = 500            #从迭代开始剪枝
        self.pruning_interval = 100             #剪枝间隔
        self.opacity_threshold_coarse = 0.005   #粗透明度阈值
        self.opacity_threshold_fine_init = 0.005    #细透明度初始阈值
        self.opacity_threshold_fine_after = 0.005   #细透明度之后阈值
        self.batch_size=1       #批量大小
        self.add_point=False    #是否添加点
        super().__init__(parser, "Optimization Parameters")

#用于合并命令行参数和配置文件中的参数
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]        #命令行中的参数
    cfgfile_string = "Namespace()"      #配置文件参数的字符串
    args_cmdline = parser.parse_args(cmdlne_string)     #解析后的命令行参数

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

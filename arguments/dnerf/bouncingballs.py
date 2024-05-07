_base_ = './dnerf_default.py'
'''
发现了这个文件继承于dnerf_default.py的参数
dnerf_default的参数在dnerf_default中
'''

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,          #定义网格的维度为2
     'input_coordinate_dim': 4,     #定义输入坐标的维度,这里理解成(x,y,z,t)
     'output_coordinate_dim': 32,   #输出坐标的维度为32
     'resolution': [64, 64, 64, 75]     #定义四种分辨率
    }
)
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.dataset import InferDatasetFromFolder, save_nii_series
from model.model import Generator

# 定义命令行参数
def args():
    parser = argparse.ArgumentParser(description='模型推断:以三维的nii文件作为输入，输出指定倍率的nii文件')
    # parser.add_argument('--batch', action="store_true", default=False, help='benchmark results path')
    parser.add_argument('--in_channels', default=1, type=int, help='影像通道，默认为灰阶影像')
    parser.add_argument('--infer_set', default='data/nii/infer', type=str, help='需推断数据集位置')
    parser.add_argument('--infer_result', default='data/infer', type=str, help='输出结果路径')
    parser.add_argument('--model_name', default='weights/netG_best_4x_performance_lossG_3.75e-03.pth',
                        type=str, help='generator model epoch name：模型名称')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor：超分辨率放大因子，默认为4')

    parser_args = parser.parse_args()

    return parser_args

# def infer_single(options):
#     # 初始化数据加载器
#     test_set = TestDatasetFromFolder(options.infer_set, upscale_factor=options.upscale_factor)
#     test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
#     test_bar = tqdm(test_loader, desc='[infer benchmark datasets]')
# 
#     # 准备输出路径
#     out_path = f"{options.infer_result}/SRF_{str(options.upscale_factor)}"
#     os.makedirs(out_path, exist_ok=True)
# 
#     # 测试循环
#     for image_name, lr_image in test_bar:
#         image_name = image_name[0]
#         with torch.no_grad():
#             lr_image = Variable(lr_image)
#             if torch.cuda.is_available():
#                 lr_image = lr_image.cuda()
# 
#         # 前向传播
#         sr_image = model(lr_image)
#     return 0
# 
# 
# def infer_batch(options):
#     # 初始化数据加载器
#     test_set = TestDatasetFromFolder(options.test_set, upscale_factor=options.upscale_factor)
#     test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
#     test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')
# 
#     # 准备输出路径
#     out_path = f"{options.infer_result}/SRF_{str(options.upscale_factor)}"
#     os.makedirs(out_path, exist_ok=True)
# 
#     # 测试循环
#     for image_name, lr_image in test_bar:
#         image_name = image_name[0]
#         with torch.no_grad():
#             lr_image = Variable(lr_image)
#             if torch.cuda.is_available():
#                 lr_image = lr_image.cuda()
# 
#         # 前向传播
#         sr_image = model(lr_image)
#     return 0


# 将主流程代码放入 if __name__ == '__main__' 块中
if __name__ == '__main__':
    # 解析命令行参数
    opt = args()
    # 初始化模型
    model = Generator(scale_factor=opt.upscale_factor, in_channels=opt.in_channels).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(opt.model_name))

    # if opt.batch:
    #     infer_batch(options=opt)
    # else:
    #     infer_single(options=opt)
    # 初始化数据加载器
    infer_set = InferDatasetFromFolder(opt.infer_set)
    infer_loader = DataLoader(dataset=infer_set, num_workers=4, batch_size=1, shuffle=False)
    infer_bar = tqdm(infer_loader, desc='[infer datasets]')

    # 准备输出路径
    out_path = f"{opt.infer_result}/SRF_{str(opt.upscale_factor)}"
    os.makedirs(out_path, exist_ok=True)

    for images, path in infer_bar:
        # 这里迭代器给出的images是tensor：[1, D, H, W], path是label的元组：(path,)
        # 如果你的getitem有两个及以上的返回值，第一个默认为data，其后的均认为是label，依次放置在元组中输出
        sr_images_list = []  # 存储所有 2D 结果的列表
        for img in images[0]: # [H, W] in [D, H, W]
            lr_image = img.unsqueeze(0).unsqueeze(0).cuda()
            with torch.no_grad():
                sr_tensor = model(lr_image)
            sr_image = sr_tensor.detach().cpu().numpy().squeeze(0).squeeze(0)  # 去掉 batch 维度 (C, H, W)

            sr_images_list.append(sr_image)  # 逐层存储

        # 整合成 3D numpy.array，沿第 0 维 (depth) 叠加
        sr_3d_array = np.stack(sr_images_list, axis=0)

        # 提取文件名
        fname = os.path.split(path[0])[-1]
        # 保存图像
        save_nii_series(image_array=sr_3d_array,
                        reference_nifti_path=path[0],
                        output_path=f"{out_path}/{fname}",
                        scale_factor=opt.upscale_factor)
import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import ssim
from utils.dataset import TestDatasetFromFolder, display_transform
from model.model import Generator

# 定义命令行参数
def args():
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    parser.add_argument('--benchmark', default='benchmark', type=str, help='benchmark results path')
    parser.add_argument('--test_set', default='data/nii/test', type=str, help='test set path')
    parser.add_argument('--test_result', default='statistics', type=str, help='test result path')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--model_name', default='weights/netG_best_4x_performance_lossG_5.08e-03.pth',
                        type=str, help='generator model epoch name')

    args = parser.parse_args()

    return args


# 将主流程代码放入 if __name__ == '__main__' 块中
if __name__ == '__main__':
    # # 添加冻结支持（如果需要生成可执行文件）
    # from multiprocessing import freeze_support
    # freeze_support()

    # 解析命令行参数
    opt = args()

    # # 初始化结果字典
    # results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
    #            'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

    # 初始化模型
    model = Generator(opt.upscale_factor).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(opt.model_name))

    # 初始化数据加载器
    test_set = TestDatasetFromFolder(opt.test_set, upscale_factor=opt.upscale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    # 准备输出路径
    out_path = f"{opt.benchmark}/SRF_{str(opt.upscale_factor)}"
    os.makedirs(out_path, exist_ok = True)

    # 测试循环
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = Variable(lr_image)
            hr_image = Variable(hr_image)
            if torch.cuda.is_available():
                lr_image = lr_image.cuda()
                hr_image = hr_image.cuda()

        # 前向传播
        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = ssim.ssim(sr_image, hr_image).item()

        # 保存结果图像
        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # # 保存 PSNR 和 SSIM
        # results[image_name.split('_')[0]]['psnr'].append(psnr)
        # results[image_name.split('_')[0]]['ssim'].append(ssim)

    # 保存统计数据
    out_path = opt.test_result
    saved_results = {'psnr': [], 'ssim': []}
    # for item in results.values():
    #     psnr = np.array(item['psnr'])
    #     ssim = np.array(item['ssim'])
    #     if (len(psnr) == 0) or (len(ssim) == 0):
    #         psnr = 'No data'
    #         ssim = 'No data'
    #     else:
    #         psnr = psnr.mean()
    #         ssim = ssim.mean()
    #     saved_results['psnr'].append(psnr)
    #     saved_results['ssim'].append(ssim)

    # data_frame = pd.DataFrame(saved_results, results.keys())
    # data_frame.to_csv(f"{opt.test_result}/srf_{str(opt.upscale_factor)}_test_results.csv", index_label='DataSet')
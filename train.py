import argparse
import os
import shutil
from math import log10
import datetime

import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.ssim import ssim
from utils.dataset import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model.loss import GeneratorLoss, TVLoss
from model.model import Generator, Discriminator


# 定义命令行参数
def args():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--batch_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
    # parser.add_argument('--experiment-start-time', type=str,
    #                     default=datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'))
    parser.add_argument('--in_channels', default=1, type=int, help='image channels, default as gray scale')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='initial learning rate of generator')
    parser.add_argument('--lr_D', default=1e-6, type=float, help='initial learning rate of discriminator')
    parser.add_argument("--milestones", default=[0.1, 0.2, 0.4], type=list, help='learning rate milestone, '
                                                                              'percentage of total epochs')
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1)
    parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
    parser.add_argument('--val_set', default='data/nii/val', type=str, help='val set path')
    parser.add_argument('--val_result', default='statistics', type=str, help='val result path')
    parser.add_argument('--train_record', default='runs', type=str, help='training records path')
    parser.add_argument('--train_set', default='data/nii/train', type=str, help='training set path')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument('--model_weights', default='weights', type=str, help='model weights path')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opt = args()
    # 全局变量定义
    NUM_CHANNELS = opt.in_channels
    TRAINING_SET = opt.train_set
    TESTING_SET = opt.val_set
    TESTING_RECORDS = opt.val_result
    RECORDING_PATH = opt.train_record
    MODEL_PATH = opt.model_weights
    # 加载数据
    train_set = TrainDatasetFromFolder(dataset_dir=f"{TRAINING_SET}",
                                       crop_size=opt.crop_size,
                                       upscale_factor=opt.upscale_factor)
    val_set = ValDatasetFromFolder(dataset_dir=f"{TESTING_SET}",
                                   upscale_factor=opt.upscale_factor)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    # 加载模型
    netG = Generator(opt.upscale_factor, in_channels=NUM_CHANNELS)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(in_channels=NUM_CHANNELS)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    # 加载损失函数
    generator_criterion = GeneratorLoss(in_channels=NUM_CHANNELS)

    # 检查cuda
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    # 加载优化函数
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_G)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_D)
    # 学习率调度器：在指定milestone，以现有的学习率乘上给定衰减因子得到新学习率
    final_milestones =[int(opt.num_epochs * p) for p in opt.milestones]
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=final_milestones, gamma=opt.lr_scheduler_gamma)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=final_milestones, gamma=opt.lr_scheduler_gamma)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, opt.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            ############################
            # (2) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D:%.4f  Loss_G:%.4f  D(x):%.4f  D(G(z)):%.4f  lr_D:%.6e  lr_G:%.6e' % (
                epoch, opt.num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes'],
                schedulerD.get_last_lr()[0], schedulerG.get_last_lr()[0]))

        netG.eval()
        out_path = f"{RECORDING_PATH}/super_resolution_factor_{str(opt.upscale_factor)}/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # 调整学习率
        schedulerG.step()
        schedulerD.step()

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valid_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valid_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valid_results['mse'] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()
                valid_results['ssims'] += batch_ssim * batch_size
                valid_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valid_results['mse'] / valid_results['batch_sizes']))
                valid_results['ssim'] = valid_results['ssims'] / valid_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valid_results['psnr'], valid_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 14)
            val_save_bar = tqdm(val_images,
                                 desc=f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} saving training results]')
            index = 1
            for image in val_save_bar:
                if index % 20 == 0:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1
                else:
                    index += 1

        # save model parameters
        torch.save(netG.state_dict(), f"{MODEL_PATH}/netG_{opt.upscale_factor}x_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"{MODEL_PATH}/netD_{opt.upscale_factor}x_epoch_{epoch}.pth")
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valid_results['psnr'])
        results['ssim'].append(valid_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = TESTING_RECORDS
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(f"{TESTING_RECORDS}/srf_{str(opt.upscale_factor)}_train_results.csv", index_label='Epoch')

            # 找到 g_loss 最小值的索引
            best_epoch = data_frame.loc[:,'Loss_G'].idxmin() - 1
            min_loss = data_frame.loc[best_epoch + 1, 'Loss_G']
            shutil.copy(f"{MODEL_PATH}/netG_{opt.upscale_factor}x_epoch_{best_epoch}.pth",
                        f"{MODEL_PATH}/netG_best_{opt.upscale_factor}x_performance_lossG_{min_loss:.2e}.pth")

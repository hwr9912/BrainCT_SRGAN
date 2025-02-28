import os
import SimpleITK as sitk
import numpy as np
from PIL import ImageDraw, ImageFont

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, GaussianBlur, ToPILImage, CenterCrop, Resize, \
    InterpolationMode, Normalize


def is_nii_file(filename: str):
    """
    判断文件是否为nii文件
    :param filename:文件名字符串
    :return:结尾为nii返回True，反之返回False
    """
    return filename.endswith('.nii')


def load_nii_image(nii_path):
    """
    使用 SimpleITK 读取 NIfTI 文件
    :param nii_path: NIfTI 文件路径
    :return: NumPy 数组
    """
    # 读取 NIfTI 文件
    image = sitk.ReadImage(nii_path)
    # 将 SimpleITK 图像转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    # 如果有多个切片，只取第一个切片
    if image_array.ndim == 3:
        image_array = image_array[0]

    image_array[image_array > 100] = 100
    image_array[image_array < 0] = 0

    return image_array / 100


# 定义一个函数，用于在图片左上角添加标记
def add_text_to_image(image, text, font_size=20, color=(255, 0, 0)):
    """
    在图片左上角添加文本标记
    :param image: PIL 图像对象
    :param text: 要添加的文本
    :param font_size: 字体大小
    :param color: 文本颜色 (R, G, B)
    :return: 添加标记后的 PIL 图像对象
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)  # 使用 Arial 字体，需要确保字体文件存在
    draw.text((5, 5), text, font=font, fill=color)  # 在左上角添加文本
    return image


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        ToTensor(),
        RandomCrop(crop_size),
        Normalize(0,1)
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_nii_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(load_nii_image(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


# 验证集加载类
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_nii_file(x)]

    def __getitem__(self, index):
        hr_image = load_nii_image(self.image_filenames[index])
        w, h = hr_image.shape
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=InterpolationMode.BICUBIC)
        hr_image = CenterCrop(crop_size)(ToPILImage()(hr_image))
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, target_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        # 修改文件过滤逻辑，只加载 .dcm 文件，并确保文件对应
        self.image_names = [x for x in os.listdir(target_dir) if is_nii_file(x)]
        self.hr_dcm_filenames = [os.path.join(target_dir, x) for x in self.image_names]
        self.lr_dcm_filenames = [os.path.join(dataset_dir, x) for x in self.image_names]

    def __getitem__(self, index):
        # 读取 DICOM 文件
        lr_image = load_nii_image(self.lr_dcm_filenames[index])
        hr_image = load_nii_image(self.hr_dcm_filenames[index])
        image_name = self.image_names[index]
        # 转换为 PyTorch 张量并添加通道维度
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)  # [H, W] -> [1, H, W]
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)  # [H, W] -> [1, H, W]
        # 生成对比用双三次插值高分辨率图像
        hr_scale = Resize(hr_image.shape[1:], interpolation=InterpolationMode.BICUBIC)  # 注意：Resize 需要整数尺寸
        hr_restore_img = hr_scale(lr_image)
        return lr_image, hr_restore_img, hr_image

    def __len__(self):
        return len(self.hr_dcm_filenames)

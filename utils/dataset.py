import os
import SimpleITK as sitk
import numpy as np
from PIL import ImageDraw, ImageFont

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode, Normalize
import torch.nn.functional as F


def is_nii_file(filename: str):
    """
    判断文件是否为nii文件
    :param filename:文件名字符串
    :return:结尾为nii返回True，反之返回False
    """
    return filename.endswith('.nii') or  filename.endswith('.nii.gz')


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


def save_nii_series(image_array, reference_nifti_path, output_path, scale_factor):
    """
    将 numpy 数组保存为 NIfTI，并使其仿射矩阵（方向）、原点和间距与参考 NIfTI 文件一致。

    参数：
        numpy_array (numpy.ndarray): 需要保存的 3D 或 4D NumPy 数组。
        reference_nifti_path (str): 参考 NIfTI 文件的路径。
        output_path (str): 输出的 NIfTI 文件路径。
    """
    # 读取参考 NIfTI 文件
    reference_img = sitk.ReadImage(reference_nifti_path)

    # 将 NumPy 数组转换为 SimpleITK 图像
    new_img = sitk.GetImageFromArray(image_array)

    # 设置仿射信息（方向、原点、间距）
    new_img.SetDirection(reference_img.GetDirection())  # 方向（旋转信息）
    new_img.SetOrigin(reference_img.GetOrigin())  # 原点（世界坐标系）
    H, W, D = reference_img.GetSpacing()
    new_img.SetSpacing((H/scale_factor, W/scale_factor, D))  # 体素间距（每个维度的尺寸）

    # 保存为 NIfTI
    sitk.WriteImage(new_img, output_path)
    # print(f"NIfTI 文件已保存至: {output_path}")


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


# 定义高斯卷积核
def gaussian_kernel(size, sigma=1.0):
    """
    生成高斯卷积核
    :param size: 卷积核大小（必须是奇数）
    :param sigma: 标准差
    :return: 高斯卷积核
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return torch.tensor(g / g.sum(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# 定义高斯平滑函数
def apply_gaussian_smoothing(image_tensor, kernel_size=5, sigma=1.0):
    """
    应用高斯平滑
    :param image_tensor: 输入图像（Tensor，形状为 [C, H, W]）
    :param kernel_size: 高斯卷积核大小
    :param sigma: 标准差
    :return: 平滑后的图像（Tensor）
    """
    # 创建高斯卷积核
    kernel = gaussian_kernel(kernel_size, sigma).to(image_tensor.device)

    # 将图像扩展为 [1, C, H, W]，以便进行卷积操作
    image_tensor = image_tensor.unsqueeze(0)

    # 应用卷积操作
    padding = kernel_size // 2
    smoothed_image = F.conv2d(image_tensor, kernel, padding=padding, groups=image_tensor.shape[1])

    # 去掉批次维度，恢复为 [C, H, W]
    smoothed_image = smoothed_image.squeeze(0)
    return smoothed_image


# 定义自定义的高斯平滑变换类
class GaussianSmoothing:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image_tensor):
        return apply_gaussian_smoothing(image_tensor, self.kernel_size, self.sigma)


# 修改后的预处理函数
def train_hr_transform(crop_size, kernel_size=5, sigma=0.5):
    return Compose([
        ToTensor(),
        GaussianSmoothing(kernel_size, sigma),  # 添加高斯平滑
        RandomCrop(crop_size)
    ])

def train_lr_transform(crop_size, upscale_factor, kernel_size=5, sigma=0.5):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        GaussianSmoothing(kernel_size, sigma)  # 添加高斯平滑
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


# 修改后的 ValDatasetFromFolder 类
class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, kernel_size=5, sigma=1.0):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_nii_file(x)]
        self.gaussian_smoothing = GaussianSmoothing(kernel_size, sigma)

    def __getitem__(self, index):
        hr_image = load_nii_image(self.image_filenames[index])
        w, h = hr_image.shape
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        # 转换为 PIL 图像
        hr_image_pil = ToPILImage()(hr_image)

        # 中心裁剪
        hr_image_cropped = CenterCrop(crop_size)(hr_image_pil)

        # 缩放为低分辨率图像
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        lr_image = lr_scale(hr_image_cropped)

        # 缩放回高分辨率图像
        hr_scale = Resize(crop_size, interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_image)

        # 转换为 Tensor
        hr_image_tensor = ToTensor()(hr_image_cropped)
        lr_image_tensor = ToTensor()(lr_image)
        hr_restore_img_tensor = ToTensor()(hr_restore_img)

        # 应用高斯平滑
        hr_image_tensor = self.gaussian_smoothing(hr_image_tensor)
        lr_image_tensor = self.gaussian_smoothing(lr_image_tensor)
        hr_restore_img_tensor = self.gaussian_smoothing(hr_restore_img_tensor)

        return lr_image_tensor, hr_restore_img_tensor, hr_image_tensor

    def __len__(self):
        return len(self.image_filenames)


# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, target_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         # 修改文件过滤逻辑，只加载 .dcm 文件，并确保文件对应
#         self.image_names = [x for x in os.listdir(target_dir) if is_nii_file(x)]
#
#     def __getitem__(self, index):
#         # 读取 DICOM 文件
#         lr_image = load_nii_image(self.lr_dcm_filenames[index])
#         hr_image = load_nii_image(self.hr_dcm_filenames[index])
#         image_name = self.image_names[index]
#         # 转换为 PyTorch 张量并添加通道维度
#         lr_image = torch.from_numpy(lr_image).unsqueeze(0)  # [H, W] -> [1, H, W]
#         hr_image = torch.from_numpy(hr_image).unsqueeze(0)  # [H, W] -> [1, H, W]
#         # 生成对比用双三次插值高分辨率图像
#         hr_scale = Resize(hr_image.shape[1:], interpolation=InterpolationMode.BICUBIC)  # 注意：Resize 需要整数尺寸
#         hr_restore_img = hr_scale(lr_image)
#         return lr_image, hr_restore_img, hr_image
#
#     def __len__(self):
#         return len(self.hr_dcm_filenames)


def load_nii_series(nii_path):
    """
    使用 SimpleITK 读取 NIfTI series文件
    :param nii_path: NIfTI 文件路径
    :return: NumPy 数组[D, W, H]，阈值截断后进行了归一化
    """
    # 读取 NIfTI 文件
    image = sitk.ReadImage(nii_path)


    # 应用高斯滤波
    gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian_filter.SetSigma(0.5)  # 设置高斯滤波的标准差
    image = gaussian_filter.Execute(image)

    # 将 SimpleITK 图像转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)

    image_array[image_array > 100] = 100
    image_array[image_array < 0] = 0

    return image_array / 100


class InferDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, kernel_size=5, sigma=1.0):
        super(InferDatasetFromFolder, self).__init__()
        # 修改文件过滤逻辑，只加载 .nii 文件，并确保文件对应
        self.image_names = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_nii_file(x)]
        self.gaussian_smoothing = GaussianSmoothing(kernel_size, sigma)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # 读取 DICOM 文件
        lr_image = load_nii_series(self.image_names[index])
        # 转换为 PyTorch 张量并添加通道维度
        if lr_image.ndim == 2:
            lr_image = np.expand_dims(lr_image, axis=0)  # [H, W] -> [1, H, W]
        return lr_image, self.image_names[index]
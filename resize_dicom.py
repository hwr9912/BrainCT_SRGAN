import os
from tqdm import tqdm
import SimpleITK as sitk

def compress_nii_files(input_dir, output_dir, compression_factor=4):
    """
    批量读取 NIfTI 文件，对前两个维度进行压缩，并将结果保存为新的 NIfTI 文件。

    参数:
        input_dir (str): 包含原始 NIfTI 文件的目录路径。
        output_dir (str): 保存压缩后的 NIfTI 文件的目录路径。
        compression_factor (int): 压缩倍数，默认为 4。
    """
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in tqdm(os.listdir(input_dir)):
        # 检查文件是否为 NIfTI 文件（通过文件扩展名）
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, filename)

            # 读取 NIfTI 文件
            image = sitk.ReadImage(file_path)

            # 获取原始图像的尺寸、原点和间距
            original_size = image.GetSize()
            original_origin = image.GetOrigin()
            original_spacing = image.GetSpacing()

            # 计算压缩后的尺寸和间距
            new_size = list(original_size)
            new_size[0] = int(original_size[0] / compression_factor)  # 压缩宽度
            new_size[1] = int(original_size[1] / compression_factor)  # 压缩高度
            new_spacing = list(original_spacing)
            new_spacing[0] *= compression_factor  # 更新宽度间距
            new_spacing[1] *= compression_factor  # 更新高度间距

            # 使用 SimpleITK 的 ResampleImageFilter 进行图像压缩
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetSize(new_size)
            resample_filter.SetOutputSpacing(new_spacing)
            resample_filter.SetOutputOrigin(original_origin)  # 保持原点不变
            resample_filter.SetInterpolator(sitk.sitkLinear)  # 使用线性插值
            compressed_image = resample_filter.Execute(image)

            # 构造输出文件路径
            output_file_path = os.path.join(output_dir, filename)

            # 保存压缩后的图像为新的 NIfTI 文件
            sitk.WriteImage(compressed_image, output_file_path)

    print("Processing complete.")


if __name__ == "__main__":
    # 示例用法
    for dataset in ["train", "test", "val"]:
        print(f"Processing {dataset} set.")
        input_directory = rf"D:\Python\computer_vision\SRGAN_dcm\data\{dataset}\target" # 替换为包含原始 DICOM 文件的目录路径
        output_directory = rf"D:\Python\computer_vision\SRGAN_dcm\data\{dataset}\data"  # 替换为保存压缩后 DICOM 文件的目录路径
        compress_nii_files(input_directory, output_directory, compression_factor=4)

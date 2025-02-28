# 颅脑CT高分辨率重建（Brain CT super-resolution）
本代码仓库相当一部分代码来源于https://github.com/leftthomas/SRGAN

训练环境依赖位于environment.yml

## 输入

输入数据已经预先将dcm的格式转为NIFTI格式（.nii），并将像素CT值截断至0-100（常见颅脑软组织CT值密度），具体的转码程序参见我的另一个[代码仓库](https://github.com/hwr9912/ToolBox.git)。

| 代码名称                        | 作用                         |
| ------------------------------- | ---------------------------- |
| WHT_convert_dicom_slices2nii.py | 将单层dicom文件转换为nii文件 |
|WHT_preprocess_nii_by_thres.py|对nii文件进行自定义阈值处理|

训练模型

```shell
python train.py
```

训练参数一览（默认值可在train.py中查阅）

```shell
options:  
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        training images crop size
  --crop_size CROP_SIZE
                        training images crop size
  --in_channels IN_CHANNELS
                        image channels, default as gray scale
  --lr_G LR_G           initial learning rate of generator
  --lr_D LR_D           initial learning rate of discriminator
  --milestones MILESTONES
                        learning rate milestone, percentage of total epochs
  --lr_scheduler_gamma LR_SCHEDULER_GAMMA
  --num_epochs NUM_EPOCHS
                        train epoch number
  --test_set TEST_SET   test set path
  --test_result TEST_RESULT
                        test result path
  --train_record TRAIN_RECORD
                        training records path
  --train_set TRAIN_SET
                        training set path
  --upscale_factor {2,4,8}
                        super resolution upscale factor
  --model_weights MODEL_WEIGHTS
                        model weights path

```

## 模型权重

由于训练数据并非源于公开数据，我无法上传原始训练数据，以下是训练数据的概览

| 集合  | 病人数 | 切片数 |
| ----- | ------ | ------ |
| train | 83     | 17662  |
| test  | 11     | 2272   |

训练过程中的参数变化上传于statistics文件夹中，供训练时参考

在完全隐去病人相关信息后，我提供了一个已训练完成的模型参数，具体文件位于weights文件夹下

## 输出

模型推断

```she
python inference.py
```

推断参数一览（默认值可在inference.py中查阅）

```she
```




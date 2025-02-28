import torch
from torch import nn
from torchvision.models.vgg import vgg16, VGG16_Weights


class GeneratorLoss(nn.Module):
    def __init__(self, in_channels):
        """
        初始化 GeneratorLoss
        :param in_channels: 输入数据的通道数
        """
        super(GeneratorLoss, self).__init__()
        # 加载预训练的 VGG16 模型并修改第一层卷积
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 修改第一层卷积的输入通道数
        vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # 使用重新定义的第一层卷积替换原 VGG 的第一层
        features = list(vgg.features)
        features[0] = vgg.features[0]
        self.loss_network = nn.Sequential(*features[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        """
        前向传播
        :param out_labels: 鉴别器输出的标签
        :param out_images: 生成器生成的图像
        :param target_images: 真实图像
        :return: 总损失
        """
        # 对抗损失
        adversarial_loss = torch.mean(1 - out_labels)
        # 感知损失
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # 图像损失
        image_loss = self.mse_loss(out_images, target_images)
        # 总变差损失 (TV Loss)
        tv_loss = self.tv_loss(out_images)
        # 总损失
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        初始化 TVLoss
        :param tv_loss_weight: TV Loss 的权重系数
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: TV Loss
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        # 计算水平和垂直方向的梯度
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        """
        计算张量的大小（不包括批量维度）
        :param t: 输入张量
        :return: 张量的大小
        """
        return t.size()[1] * t.size()[2] * t.size()[3]
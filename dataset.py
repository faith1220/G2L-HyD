import random  # 导入随机数模块

from torchvision import transforms  # 导入Pytorch的图像预处理库
from PIL import Image  # 用于图像打开和处理
import os  # 文件路径处理
import torch  # PyTorch深度学习库
import glob  # 文件模式匹配
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder  # 常见图像数据集接口
import numpy as np  # 科学计算库
import torch.multiprocessing  # 多进程支持
import json  # 用于读取json文件
from torch.utils.data import Dataset

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')  # 设置数据加载时的多进程共享策略为文件系统

# 定义标准数据预处理方法
def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train  # 默认使用 ImageNet 均值
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train  # 默认使用 ImageNet 标准差
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 PyTorch 张量并归一化到 [0,1]
        transforms.CenterCrop(isize),  # 从中心裁剪到指定大小
        transforms.Normalize(mean=mean_train, std=std_train)])  # 使用给定均值方差进行标准化
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),  # 调整标签图像大小
        transforms.CenterCrop(isize),  # 中心裁剪
        transforms.ToTensor()])  # 转换为张量(不做归一化)
    return data_transforms, gt_transforms  # 返回图像和标签的 transform 对象

# 定义强数据增强方法（随机裁剪、翻转、颜色扰动）
def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),  # 调整大小
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(0.1, 0.1, 0.1),  # 随机色彩抖动
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=mean_train, std=std_train)])  # 标准化
    return data_transforms  # 返回数据增强对象

# 以下类定义了多种工业异常检测数据集的加载与处理逻辑
# 包括 MVTecDataset, RealIADDataset, LOCODataset, InsPLADDataset, AeBADDataset, MiniDataset, MVTecDRAEMDataset, MVTecSimplexDataset
# 每个类都继承自 torch.utils.data.Dataset，主要步骤：
# 1. 构造函数中定义路径、transform 以及加载样本路径。
# 2. load_dataset() 方法读取文件路径并分类标签。
# 3. __getitem__() 方法打开图像并返回 (图像张量, 掩码, 标签, 图像路径)。
# 4. __len__() 方法返回样本数量。
#
# 每个类根据不同数据集结构略有差异：
# - MVTecDataset: 经典 MVTec-AD 数据集，包含 good 和 defect 图像以及掩码。
# - RealIADDataset: 使用 json 文件记录路径信息的 Real-IAD 数据集。
# - LOCODataset: 类似 MVTec，但掩码路径结构不同。
# - InsPLADDataset: 工业零件异常数据集，仅提供标签。
# - AeBADDataset: 带有多域数据的工业数据集。
# - MiniDataset: 简单文件夹图像加载器。
# - MVTecDRAEMDataset: 用于 DRAEM 异常合成的训练数据集。
# - MVTecSimplexDataset: 用于添加 Simplex 噪声异常样本的训练集。

# ========== MVTecDataset 类逐行注释 ==========
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        # 根据 phase 设置不同的图像路径
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')  # 训练集路径
        else:
            self.img_path = os.path.join(root, 'test')  # 测试集图像路径
            self.gt_path = os.path.join(root, 'ground_truth')  # 测试集对应掩码路径
        self.transform = transform  # 图像预处理
        self.gt_transform = gt_transform  # 掩码预处理
        # 加载所有图像和标签路径
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0  # 类别索引(默认0)

    def load_dataset(self):
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = [], [], [], []
        defect_types = os.listdir(self.img_path)  # 获取所有缺陷类型文件夹
        for defect_type in defect_types:
            if defect_type == 'good':  # 正常样本
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))  # good 样本无掩码，用0占位
                tot_labels.extend([0] * len(img_paths))  # 标签=0
                tot_types.extend(['good'] * len(img_paths))
            else:  # 异常样本
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort(); gt_paths.sort()  # 排序保证一一对应
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))  # 标签=1
                tot_types.extend([defect_type] * len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "测试图像和掩码数量不一致"
        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)  # 返回数据量

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')  # 打开图像
        img = self.transform(img)  # 预处理
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])  # 正常样本无掩码，创建全零
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)  # 掩码预处理
        assert img.size()[1:] == gt.size()[1:], "图像与掩码尺寸不匹配"  
        return img, gt, label, img_path  # 返回图像、掩码、标签和路径


# ========== MVTecDataset 类逐行注释 ==========
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')  # 训练集路径
        else:
            self.img_path = os.path.join(root, 'test')  # 测试图像路径
            self.gt_path = os.path.join(root, 'ground_truth')  # 测试掩码路径
        self.transform = transform  # 图像预处理方法
        self.gt_transform = gt_transform  # 掩码预处理方法
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # 加载数据集
        self.cls_idx = 0

    def load_dataset(self):
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = [], [], [], []
        defect_types = os.listdir(self.img_path)  # 列出类别文件夹
        for defect_type in defect_types:
            if defect_type == 'good':  # 正常类别
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))  # good 无掩码
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:  # 异常类别
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort(); gt_paths.sort()  # 排序保证对齐
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "测试图像与掩码数量不一致"
        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])  # 正常样本生成全零mask
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "图像与掩码尺寸不匹配"
        return img, gt, label, img_path
    
# 原本的操作
# ========== RealIADDataset 类 ==========
class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)  # 图像根目录
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')  # json注释文件
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)
        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []
        data_set = class_json[phase]  # 根据训练或测试选择子集
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'  # 标签判断
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])
        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:], "图像与掩码尺寸不匹配"
        return img, gt, label, img_path

class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise

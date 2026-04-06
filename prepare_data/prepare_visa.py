# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# 导入必需的库
import argparse  # 用于解析命令行参数
import os        # 用于操作系统相关功能，如文件路径操作
import shutil    # 用于文件复制等高级文件操作
import csv       # 用于读取CSV文件
from PIL import Image    # Python图像库，用于图像处理
import numpy as np       # 数值计算库，用于数组操作


def _mkdirs_if_not_exists(path):
    """如果目录不存在则创建目录的辅助函数"""
    if not os.path.exists(path):  # 检查路径是否存在
        os.makedirs(path)         # 如果不存在则递归创建目录


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Data preparation')  # 创建参数解析器，描述为"数据准备"

# 添加split-type参数：数据分割类型，默认为'1cls'（单类别），可选择1cls、2cls_highshot、2cls_fewshot
parser.add_argument('--split-type', default='1cls', type=str, help='1cls, 2cls_highshot, 2cls_fewshot')

# 添加data-folder参数：指定下载的VisA数据集路径
parser.add_argument('--data-folder', default='/mnt/sdb/huoyongzhen/Dinomaly/VisA', type=str,
                    help='the path to downloaded VisA dataset')

# 添加save-folder参数：指定重新组织后的数据集保存路径
parser.add_argument('--save-folder', default='/mnt/sdb/huoyongzhen/Dinomaly/data/VisA_pytorch/', type=str,
                    help='the target path to save the reorganized VisA dataset facilitating data loading in pytorch')

# 添加split-file参数：指定用于分割数据集的CSV文件路径
parser.add_argument('--split-file', default='/mnt/sdb/huoyongzhen/Dinomaly/VisA/split_csv/1cls.csv', type=str,
                    help='the csv file to split downloaded VisA dataset')
# 解析命令行参数
config = parser.parse_args()

# 从配置中提取参数值
split_type = config.split_type      # 获取分割类型
split_file = config.split_file      # 获取分割文件路径
data_folder = config.data_folder    # 获取原始数据文件夹路径
save_folder = os.path.join(config.save_folder, split_type)  # 构建保存路径，加上分割类型子目录

# 定义VisA数据集中包含的所有数据类别列表
data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']

# 根据分割类型处理数据
if split_type == '1cls':  # 如果是单类别分割（只有训练集用正常样本，测试集有正常和异常样本）
    # 为每个数据类别创建目录结构
    for data in data_list:  # 遍历每个数据类别
        # 定义各个文件夹路径
        train_folder = os.path.join(save_folder, data, 'train')         # 训练集文件夹
        test_folder = os.path.join(save_folder, data, 'test')           # 测试集文件夹
        mask_folder = os.path.join(save_folder, data, 'ground_truth')   # 真值掩码文件夹

        # 定义具体的子文件夹路径
        train_img_good_folder = os.path.join(train_folder, 'good')      # 训练集正常图像文件夹
        test_img_good_folder = os.path.join(test_folder, 'good')        # 测试集正常图像文件夹
        test_img_bad_folder = os.path.join(test_folder, 'bad')          # 测试集异常图像文件夹
        test_mask_bad_folder = os.path.join(mask_folder, 'bad')         # 测试集异常图像的掩码文件夹

        # 创建所有必需的目录
        _mkdirs_if_not_exists(train_img_good_folder)   # 创建训练集正常图像目录
        _mkdirs_if_not_exists(test_img_good_folder)    # 创建测试集正常图像目录
        _mkdirs_if_not_exists(test_img_bad_folder)     # 创建测试集异常图像目录
        _mkdirs_if_not_exists(test_mask_bad_folder)    # 创建测试集异常掩码目录

    # 打开并读取CSV分割文件
    with open(split_file, 'r') as file:  # 以读模式打开CSV文件
        csvreader = csv.reader(file)     # 创建CSV读取器
        header = next(csvreader)         # 跳过CSV文件的标题行
        
        # 逐行处理CSV文件中的数据
        for row in csvreader:  # 遍历CSV文件的每一行
            # 解析CSV行数据：对象类别、数据集类型、标签、图像路径、掩码路径
            object, set, label, image_path, mask_path = row
            
            # 标准化标签名称
            if label == 'normal':  # 如果标签是'normal'
                label = 'good'     # 改为'good'
            else:                  # 否则
                label = 'bad'      # 改为'bad'
            
            # 从完整路径中提取文件名
            image_name = image_path.split('/')[-1]  # 提取图像文件名
            mask_name = mask_path.split('/')[-1]    # 提取掩码文件名
            
            # 构建源文件和目标文件的完整路径
            img_src_path = os.path.join(data_folder, image_path)                      # 源图像文件路径
            msk_src_path = os.path.join(data_folder, mask_path)                       # 源掩码文件路径
            img_dst_path = os.path.join(save_folder, object, set, label, image_name)  # 目标图像文件路径
            msk_dst_path = os.path.join(save_folder, object, 'ground_truth', label, mask_name)  # 目标掩码文件路径
            
            # 复制图像文件到目标位置
            shutil.copyfile(img_src_path, img_dst_path)  # 复制图像文件
            
            # 只对测试集中的异常样本处理掩码
            if set == 'test' and label == 'bad':  # 如果是测试集的异常样本
                mask = Image.open(msk_src_path)   # 打开掩码图像

                # 二值化掩码处理
                mask_array = np.array(mask)        # 将PIL图像转换为numpy数组
                mask_array[mask_array != 0] = 255  # 将所有非零像素设置为255（白色）
                mask = Image.fromarray(mask_array) # 将numpy数组转换回PIL图像

                mask.save(msk_dst_path)           # 保存处理后的掩码图像
else:  # 如果是2类别分割（训练集和测试集都包含正常和异常样本）
    # 为每个数据类别创建更复杂的目录结构
    for data in data_list:  # 遍历每个数据类别
        # 定义主要文件夹路径
        train_folder = os.path.join(save_folder, data, 'train')         # 训练集文件夹
        test_folder = os.path.join(save_folder, data, 'test')           # 测试集文件夹
        mask_folder = os.path.join(save_folder, data, 'ground_truth')   # 真值掩码文件夹
        train_mask_folder = os.path.join(mask_folder, 'train')          # 训练集掩码文件夹
        test_mask_folder = os.path.join(mask_folder, 'test')            # 测试集掩码文件夹

        # 定义图像子文件夹路径
        train_img_good_folder = os.path.join(train_folder, 'good')      # 训练集正常图像
        train_img_bad_folder = os.path.join(train_folder, 'bad')        # 训练集异常图像
        test_img_good_folder = os.path.join(test_folder, 'good')        # 测试集正常图像
        test_img_bad_folder = os.path.join(test_folder, 'bad')          # 测试集异常图像

        # 定义掩码子文件夹路径
        train_mask_bad_folder = os.path.join(train_mask_folder, 'bad')  # 训练集异常掩码
        test_mask_bad_folder = os.path.join(test_mask_folder, 'bad')    # 测试集异常掩码

        # 创建所有必需的目录
        _mkdirs_if_not_exists(train_img_good_folder)   # 创建训练集正常图像目录
        _mkdirs_if_not_exists(train_img_bad_folder)    # 创建训练集异常图像目录
        _mkdirs_if_not_exists(test_img_good_folder)    # 创建测试集正常图像目录
        _mkdirs_if_not_exists(test_img_bad_folder)     # 创建测试集异常图像目录
        _mkdirs_if_not_exists(train_mask_bad_folder)   # 创建训练集异常掩码目录
        _mkdirs_if_not_exists(test_mask_bad_folder)    # 创建测试集异常掩码目录

    # 打开并读取CSV分割文件
    with open(split_file, 'r') as file:  # 以读模式打开CSV文件
        csvreader = csv.reader(file)     # 创建CSV读取器
        header = next(csvreader)         # 跳过CSV文件的标题行
        
        # 逐行处理CSV文件中的数据
        for row in csvreader:  # 遍历CSV文件的每一行
            # 解析CSV行数据
            object, set, label, image_path, mask_path = row
            
            # 标准化标签名称
            if label == 'normal':  # 如果标签是'normal'
                label = 'good'     # 改为'good'
            else:                  # 否则
                label = 'bad'      # 改为'bad'
            
            # 从完整路径中提取文件名
            image_name = image_path.split('/')[-1]  # 提取图像文件名
            mask_name = mask_path.split('/')[-1]    # 提取掩码文件名
            
            # 构建源文件和目标文件的完整路径
            img_src_path = os.path.join(data_folder, image_path)                           # 源图像文件路径
            msk_src_path = os.path.join(data_folder, mask_path)                            # 源掩码文件路径
            img_dst_path = os.path.join(save_folder, object, set, label, image_name)       # 目标图像文件路径
            msk_dst_path = os.path.join(save_folder, object, 'ground_truth', set, label, mask_name)  # 目标掩码文件路径
            
            # 复制图像文件到目标位置
            shutil.copyfile(img_src_path, img_dst_path)  # 复制图像文件
            
            # 对所有异常样本处理掩码（不限于测试集）
            if label == 'bad':                    # 如果是异常样本
                mask = Image.open(msk_src_path)   # 打开掩码图像

                # 二值化掩码处理
                mask_array = np.array(mask)        # 将PIL图像转换为numpy数组
                mask_array[mask_array != 0] = 255  # 将所有非零像素设置为255（白色）
                mask = Image.fromarray(mask_array) # 将numpy数组转换回PIL图像

                mask.save(msk_dst_path)           # 保存处理后的掩码图像

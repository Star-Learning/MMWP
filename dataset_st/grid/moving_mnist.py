import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    def __init__(self, root, train=True, split_ratio=10, transform=None):
        """
        初始化 MovingMNIST 数据集。

        参数:
        - root (str): 数据集根目录，包含 mnist_test_seq.npy 文件。
        - train (bool): 是否为训练集。True 表示训练集，False 表示测试集。
        - split_ratio (int): 用于划分训练和测试的帧数比例。
        - transform (callable, optional): 应用于每帧的转换函数。
        """
        self.root = root
        self.train = train
        self.split_ratio = split_ratio
        self.transform = transform

        # 加载数据
        path = os.path.join(root, 'mnist_test_seq.npy')
        data = np.load(path)  # 形状: (20, 10000, 64, 64)
        data = np.transpose(data, (1, 0, 2, 3))  # 转换为 (10000, 20, 64, 64)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据序列。

        返回:
        - frames (Tensor): 形状为 (T, 1, H, W) 的张量，T 为帧数。
        """
        sequence = self.data[idx]  # 形状: (20, 64, 64)

        # 根据训练或测试划分帧
        if self.train:
            frames = sequence[:self.split_ratio]
        else:
            frames = sequence[self.split_ratio:]

        # 添加通道维度
        frames = frames[:, np.newaxis, :, :]  # 形状: (T, 1, 64, 64)

        # 转换为张量并归一化
        frames = torch.from_numpy(frames).float() / 255.0

        # 应用转换（如果有）
        if self.transform:
            frames = self.transform(frames)

        return frames
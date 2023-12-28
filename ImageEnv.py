import gym as gym
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os
import random
import numpy as np
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=ToTensor()):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = sorted(self.get_image_files())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path, 0)
        if self.transform:
            image = self.transform(image)
        return image

    def get_image_files(self):
        image_files = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_files.append(os.path.join(root, file))
        return image_files


def getPSNR(gt,image):
    # 假设 gt 和 image 是相同形状的 NumPy 数组，取值范围是 [0, 1]
    # 计算均方误差（MSE）
    mse = np.mean(np.square(gt - image))
    # 计算 PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    # print("PSNR:", psnr)
    return psnr
class ImageDenoisingEnv(gym.Env):
    def __init__(self):
        # 加载图像数据集
        self.dataset = CustomImageDataset(folder_path= '../BSD68/cropped/clean')
        self.inputs = CustomImageDataset(folder_path= '../BSD68/cropped/noise')
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,63,63), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=7, shape=(1,63,63), dtype=np.uint8)
        self._max_episode_steps = 5

    def step(self, action):
        # action.shape=[1,8,70,70]
        last_observe = self.observation.copy()

        moved_add = self.observation + 1.0/255.0
        moved_neg = self.observation - 1.0/255.0
        gaussian = np.zeros(self.observation.shape, np.float32)
        gaussian2 = np.zeros(self.observation.shape, np.float32)
        bilateral = np.zeros(self.observation.shape, np.float32)
        bilateral2 = np.zeros(self.observation.shape, np.float32)
        median = np.zeros(self.observation.shape, np.float32)
        box = np.zeros(self.observation.shape, np.float32)

        # if np.sum(action == 0) > 0:
        gaussian[0] = cv2.GaussianBlur(self.observation[0], ksize=(5, 5), sigmaX=0.5)
        # if np.sum(action == 1) > 0:
        gaussian2[0] = cv2.GaussianBlur(self.observation[0], ksize=(5, 5), sigmaX=1.5)
        # if np.sum(action == 2) > 0:
        bilateral[0] = cv2.bilateralFilter(self.observation[0], d=5, sigmaColor=0.1, sigmaSpace=5)
        # if np.sum(action == 3) > 0:
        bilateral2[0] = cv2.bilateralFilter(self.observation[0], d=5, sigmaColor=1.0, sigmaSpace=5)
        # if np.sum(action == 4) > 0:
        median[0] = cv2.medianBlur(self.observation[0], ksize=5)
        # if np.sum(action == 5) > 0:
        box[0] = cv2.boxFilter(self.observation[0], ddepth=-1, ksize=(5, 5))

        # 执行去噪操作,# 更新状态
        # denoised_image = action * self.observation
        self.observation = np.where(action == 0, gaussian, self.observation)
        self.observation = np.where(action == 1, gaussian2, self.observation)
        self.observation = np.where(action == 2, bilateral, self.observation)
        self.observation = np.where(action == 3, bilateral2, self.observation)
        self.observation = np.where(action == 4, median, self.observation)
        self.observation = np.where(action == 5, box, self.observation)
        self.observation = np.where(action == 6, moved_neg, self.observation)
        self.observation = np.where(action == 7, moved_add, self.observation)

        # 计算奖励
        reward = getPSNR(self.gt,self.observation)- getPSNR(self.gt,last_observe)
        # reward = (np.abs(last_observe - self.observation))*

        # 返回下一个观察值、奖励、是否终止以及其他信息
        return self.observation, reward, False, {}

    def reset(self):
        # 重置环境并返回初始观察值
        # 随机选择图像
        random_index = random.randint(0, len(self.dataset) - 1)
        # 根据索引获取图像和标签
        imageGT = self.dataset[random_index]
        image = self.inputs[random_index]
        # 将图像作为观察值
        self.observation = image.numpy()
        self.gt = imageGT.numpy()

        return self.observation

import os
import random
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image
import cv2


class TrainDataSet(Dataset):  # 训练数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=256):
        super(TrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        try:
            inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
            # 在这里读取数据的操作
        except Exception as e:
            print(f"An error occurred: {e}. Returning data from index-1.")
            if index > 0:
                inputImagePath = os.path.join(self.inputPath, self.inputImages[index - 1])
            else:
                print("Index out of range. Cannot return data.")
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片
        # print(inputImagePath)

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        # print(targetImagePath)
        input_path = self.inputImages[index]
        file_name = os.path.basename(input_path)

        # 提取文件名前缀（如 0025）
        file_prefix = file_name.split('_')[0]
        gt_path = f"/home/liu/jxl/misttrain_16_256/hr_256/{file_prefix}.png"



        targetImage= Image.open(gt_path).convert("RGB")
        hh, ww = targetImage.size  # 图片的宽和高
        # input_ = inputImage.crop((cc, rr, cc + ps, rr + ps))  # 裁剪 patch ，输入和目标 patch 要对应相同
        # target = targetImage.crop((cc, rr, cc + ps, rr + ps))
        if hh > ps and ww > ps:
            rr = random.randint(0, hh - ps)  # 随机数：patch 左上角的坐标 (rr, cc)
            cc = random.randint(0, ww - ps)
            input_ = inputImage.crop((cc, rr, cc + ps, rr + ps))  # 裁剪 patch ，输入和目标 patch 要对应相同
            target = targetImage.crop((cc, rr, cc + ps, rr + ps))
        else:
            input_ = inputImage.resize((256, 256))
            target = targetImage.resize((256, 256))

        input_ = ttf.to_tensor(input_)  # 将图片转为张量
        target = ttf.to_tensor(target)



        return input_, target


class ValueDataSet(Dataset):  # 训练数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=256):
        super(ValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        try:
            inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
            # 在这里读取数据的操作
        except Exception as e:
            print(f"An error occurred: {e}. Returning data from index-1.")
            if index > 0:
                inputImagePath = os.path.join(self.inputPath, self.inputImages[index - 1])
            else:
                print("Index out of range. Cannot return data.")
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片
        # print(inputImagePath)

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        # print(targetImagePath)
        input_path = self.inputImages[index]
        file_name = os.path.basename(input_path)

        # 提取文件名前缀（如 0025）
        file_prefix = file_name.split('_')[0]
        gt_path = f"/home/liu/jxl/misttrain_16_256/hr_256/{file_prefix}.png"



        targetImage= Image.open(gt_path).convert("RGB")
        hh, ww = targetImage.size  # 图片的宽和高
        # input_ = inputImage.crop((cc, rr, cc + ps, rr + ps))  # 裁剪 patch ，输入和目标 patch 要对应相同
        # # target = targetImage.crop((cc, rr, cc + ps, rr + ps))
        # if hh > ps and ww > ps:
        #     rr = random.randint(0, hh - ps)  # 随机数：patch 左上角的坐标 (rr, cc)
        #     cc = random.randint(0, ww - ps)
        #     input_ = inputImage.crop((cc, rr, cc + ps, rr + ps))  # 裁剪 patch ，输入和目标 patch 要对应相同
        #     target = targetImage.crop((cc, rr, cc + ps, rr + ps))
        # else:
        #     input_ = inputImage.resize((ps, ps))
        #     target = targetImage.resize((ps, ps))

        input_ = ttf.to_tensor(inputImage)  # 将图片转为张量
        target = ttf.to_tensor(targetImage)



        return input_, target


class TestDataSet(Dataset):  # 测试数据集
    def __init__(self, inputPathTest):
        super(TestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)  # 输入图片路径下的所有文件名列表

    def __len__(self):
        return len(self.inputImages)  # 路径里的图片数量

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片

        input_ = ttf.to_tensor(inputImage)  # 将图片转为张量

        return input_,self.inputImages[index]


class ValueDataSet2(Dataset):  # 评估数据集
    def __init__(self, inputPathTrain, targetPathTrain):
        super(ValueDataSet2, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        # self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):
        # ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片,灰度图

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')
        targetImage = targetImage.resize((256, 256))
        inputImage = inputImage.resize((256, 256))
        # inputImage = ttf.center_crop(inputImage, (ps, ps))
        # targetImage = ttf.center_crop(targetImage, (ps, ps))

        input_ = ttf.to_tensor(inputImage)  # 将图片转为张量
        target = ttf.to_tensor(targetImage)

        return input_, target


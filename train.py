import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from PIL import Image
from ResNet34 import MyResNetD_Plus
import numpy as np

batch_size = 2
learning_rate = 1e-4
epochs = 50
train_path = r'E:\python\pycharm\projects\octa\pic_sort\train'
val_path = r'E:\python\pycharm\projects\octa\pic_sort\test'


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, labels):
        super().__init__()
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


def tran(folder_path):
    ret = []
    lab = []
    # 步骤1：列出文件夹中的所有文件
    file_list = os.listdir(folder_path)
    # 步骤2：筛选出图像文件
    for file_case in file_list:
        for root, dirs, files in os.walk(os.path.join(folder_path, file_case)):
            for dir in dirs:
                image_paths = []
                file_path = os.path.join(root, dir)
                for r, d, fs in os.walk(file_path):
                    for f in fs:
                        if f.endswith(".png"):
                            image_paths.append([os.path.join(file_path, f)])

                # 步骤4-5：打开图像、转换为灰度图像
                gray_images = []
                for image_path in image_paths:
                    image = Image.open(image_path[0])
                    gray_image = image.convert('L')  # 转换为灰度图像
                    gray_images.append(gray_image)

                # 步骤6：将灰度图像转换为Tensor对象
                tensor_images = [torchvision.transforms.ToTensor()(gray_image) for gray_image in gray_images]
                combined_tensor = torch.cat(tensor_images, dim=0)  # 在通道维度上合并张量
                ret.append(combined_tensor)
                lab.append(int(file_case))

    ret1 = []
    lab1 = []
    for i in range(len(ret)):
        if ret[i].shape == torch.Size([5, 1024, 1024]):
            ret1.append(ret[i])
            lab1.append(lab[i])

    return ret1, lab1


data, label = tran(train_path)  # data: 126,[5,1024,1024] leb:133]
# v_data, v_label = tran(val_path)

data1 = data[:5]
label1 = label[:5]

# 创建Dataset类实例
trainData = MyDataset(tensors=data1, labels=label1)
# valData = MyDataset(tensors=v_data, labels=v_label)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=5)
# valLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False, num_workers=5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("\033[34m当前计算设备为{}\033[0m".format(device))

model = MyResNetD_Plus(6)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    total_loss = 0
    train_corrects = 0
    for i, (image, label) in enumerate(trainLoader):
        image = image.to(device)  # 同理
        label = label.to(device)  # 同理
        optimizer.zero_grad()
        target = model(image)

        loss = criterion(target, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        max_value, max_index = torch.max(target, 1)
        pred_label = max_index.cpu().numpy()
        true_label = label.cpu().numpy()
        train_corrects += np.sum(pred_label == true_label)

    return total_loss / float(len(trainLoader)), train_corrects / 125


if __name__ == '__main__':
    # 训练模型
    epoch_list = []
    acc_list = []
    train()

    plt.plot(epoch_list, acc_list, color='red')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

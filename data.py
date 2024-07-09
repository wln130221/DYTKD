from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 将数据集图片放缩成224*224大小
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    "test": transforms.Compose([transforms.Resize((224, 224)),  # 调整图片大小为224*224
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# (2)、加载数据集

train_dataset = torchvision.datasets.ImageFolder(root="./dataset/enhance_Rice Leaf Disease Images/train",
                                                 transform=data_transform["train"])
print(train_dataset.class_to_idx)  # 获取训练集中不同类别对应的索引序号

test_dataset = torchvision.datasets.ImageFolder(root="./dataset/enhance_Rice Leaf Disease Images/test",
                                                transform=data_transform["test"])
print(test_dataset.class_to_idx)  # 获取测试集中不同类别对应的索引序号
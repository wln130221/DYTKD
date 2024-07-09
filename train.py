import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset.data import train_dataset, test_dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义自定义的预训练权重路径
teacher_pretrained_path = './resnet152-b121ed2d.pth'
# 定义教师模型 ResNet-34，加载自定义的预训练权重
teacher_model = models.resnet152(pretrained=False)  # 不使用默认的预训练权重
state_dict = torch.load(teacher_pretrained_path)
teacher_model.load_state_dict(state_dict)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 4)  # 输出改为4个类别
teacher_model = teacher_model.to(device)

# 定义学生模型 ResNet-18，不进行预训练
student_model = models.resnet34(pretrained=False)
student_model.fc = nn.Linear(student_model.fc.in_features, 4)  # 输出改为4个类别
student_model = student_model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 定义优化器和损失函数
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
criterion_ce = nn.CrossEntropyLoss()

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader.dataset), 100. * correct / total


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # TN, FP, FN, TP = conf_matrix.ravel()
    tp = conf_matrix.diagonal()
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = conf_matrix.sum() - (fp + fn + tp)
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    # 计算指标
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = 100. * correct / total
    precision = 100. * precision_score(all_labels, all_preds, average='macro')
    recall = 100.* recall_score(all_labels, all_preds, average='macro')
    # print('Precision: {precision:.2f}, Recall: {recall:.2f}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
    return test_loss, test_acc,precision,recall


# 训练教师模型并输出准确率
print("Training Teacher Model...")
for epoch in range(10):
    train_loss, train_acc = train(teacher_model, train_loader, optimizer_teacher, criterion_ce, device)
    test_loss, test_acc,precision,recall= test(teacher_model, test_loader, criterion_ce, device)
    print(
        f"Teacher Model - Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%,Precision: {precision:.2f}, Recall: {recall:.2f}")

# 训练学生模型并输出准确率
print("Training Student Model...")
for epoch in range(10):
    train_loss, train_acc = train(student_model, train_loader, optimizer_student, criterion_ce, device)
    test_loss, test_acc ,precision,recall= test(student_model, test_loader, criterion_ce, device)
    print(
        f"Student Model - Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%,Precision: {precision:.2f}, Recall: {recall:.2f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset.data import train_dataset,test_dataset
import csv
from sklearn.metrics import confusion_matrix,precision_score,recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# 定义损失函数
# class KD_loss(nn.Module):
#     def __init__(self, temperature=3):
#         super(KD_loss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def forward(self, student_outputs, teacher_outputs, labels):
#         soft_teacher_outputs = nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
#         soft_student_outputs = nn.functional.log_softmax(student_outputs / self.temperature, dim=1)
#         kd_loss = nn.functional.kl_div(soft_student_outputs, soft_teacher_outputs, reduction='batchmean') * (
#                     self.temperature ** 2)
#         ce_loss = self.cross_entropy(student_outputs, labels)
#         return kd_loss

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3, alpha=0.4):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.criterion_ce(student_logits, labels)
        kl_loss = self.criterion_kl(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义自定义的预训练权重路径
teacher_pretrained_path = './resnet152-b121ed2d.pth'
# 定义教师模型 ResNet-34，加载自定义的预训练权重
teacher_model = models.resnet152(pretrained=False)  # 不使用默认的预训练权重
state_dict = torch.load(teacher_pretrained_path)
teacher_model.load_state_dict(state_dict)
# num_ftrs = teacher_model.fc.in_features
# teacher_model.fc = nn.Linear(num_ftrs, 4)
teacher_model.eval()
teacher_model = teacher_model.to(device)

# 定义学生模型 ResNet-18，不进行预训练
student_model = models.resnet34(pretrained=False)
# student_model.fc = nn.Linear(num_ftrs, 4)  # 输出改为4个类别
student_model = student_model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 定义优化器和损失函数
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = DistillationLoss(temperature=3)


# 训练循环
num_epochs = 10
temperature=3
alpha=0.4
distillation_loss = DistillationLoss()
# temperature_adjustment_factor = 0.1
with open('15234.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy'])
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # criterion_ce = nn.CrossEntropyLoss()
            # loss_ce = criterion_ce(student_outputs, labels)
            # if loss_ce > 0.5:  # 举例，根据具体情况调整阈值
            #     temperature *= (1 + temperature_adjustment_factor)  # 增大温度参数
            # elif loss_ce < 0.2:
            #     temperature *= (1 - temperature_adjustment_factor)  # 减小温度参数
            # logits_teacher_soft = teacher_outputs / temperature
            # logits_student_soft = student_outputs / temperature

            # loss_kd = criterion(student_outputs,teacher_outputs, labels)
            # loss = (1 - alpha) *loss_kd + alpha * loss_ce
            loss = distillation_loss(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        writer.writerow([epoch + 1, epoch_loss, train_acc])

student_model.eval()
test_loss = 0.0
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = student_model(inputs)
        criterion_ce = nn.CrossEntropyLoss()
        loss_ce = criterion_ce(outputs, labels)
        test_loss += loss_ce.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_acc = 100. * correct / total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix')
print(conf_matrix)

tp = conf_matrix.diagonal()
fp = conf_matrix.sum(axis=0) - tp
fn = conf_matrix.sum(axis=1) - tp
tn = conf_matrix.sum() - (fp + fn + tp)
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
precision = 100. * precision_score(all_labels, all_preds, average='macro')
recall = 100.* recall_score(all_labels, all_preds, average='macro')
print(f'precision: {precision:.2f}, recall: {recall:.2f}%')
# 显示混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.title('Confusion Matrix')
plt.show()

# 分类报告
# class_report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
# print('Classification Report')
# print(class_report)
# 保存训练好的学生模型
torch.save(student_model.state_dict(), 'resnet18_distilled.pth')



import pandas as pd
import matplotlib.pyplot as plt

# # 读取教师数据的CSV文件
# teacher_data = pd.read_csv('teacher.csv')
# teacher_acc = teacher_data['Train Accuracy']
# teacher_loss = teacher_data['Train Loss']
#
# # 读取学生数据的CSV文件
# student_data = pd.read_csv('student.csv')
# student_acc = student_data['Train Accuracy']
# student_loss = student_data['Train Loss']
# print(student_acc)
# # 读取kd数据的CSV文件
# kd_data = pd.read_csv('数据增强后.csv')
# kd_acc = kd_data['Train Accuracy']
# kd_loss = kd_data['Train Loss']
# # print()
#
# teacher_index = range(len(teacher_acc))
# student_index = range(len(student_acc))
# kd_index = range(len(kd_acc))
#
# # # 绘制教师数据的准确率
# plt.plot(teacher_index, teacher_acc, label='Teacher', color='r', linestyle='-')
# # # 绘制学生数据的准确率
# plt.plot(student_index, student_acc, label='Student', color='g', linestyle='--')
# # # 绘制kd数据的准确率
# plt.plot(kd_index, kd_acc, label='KD', color='b', linestyle=':')
#
# # 添加图例、标题和坐标轴标签
# plt.legend()
# #plt.title('Accuracy Comparison of Teacher, Student, and KD')
# plt.xlabel('epoch')
# plt.ylabel('Accuracy (acc)')
#
# # 显示图表
# plt.show()
# #
# #
# # plt.plot(teacher_index, teacher_loss, label='Teacher', color='r', linestyle='-')
# # # 绘制学生数据的准确率
# # plt.plot(student_index, student_loss, label='Student', color='g', linestyle='--')
# # # 绘制kd数据的准确率
# # plt.plot(kd_index, kd_loss, label='KD', color='b', linestyle=':')
# #
# # # 添加图例、标题和坐标轴标签
# # plt.legend()
# # # plt.title('loss Comparison of Teacher, Student, and KD')
# # plt.xlabel('epoch')
# # plt.ylabel('loss')
# #
# # plt.show()
#
#
#
#
#
# # 读取kd数据的CSV文件
# original_kd_data = pd.read_csv('数据增强前kd.csv')
# # 假设准确率（acc）在'accuracy'列中
# original_kd_acc = original_kd_data['Train Accuracy']
# # kd_loss = kd_data['Train Loss']
#
# kd_data = pd.read_csv('数据增强后.csv')
# # 假设准确率（acc）在'accuracy'列中
# kd_acc = kd_data['Train Accuracy']
# # kd_loss = kd_data['Train Loss']
#
# # 如果CSV文件中包含x轴的数据（例如迭代次数），请相应地读取
# # 这里假设没有x轴数据，所以我们创建一个简单的索引
# original_kd_index = range(len(original_kd_acc))
# kd_index = range(len(kd_acc))
#
# import matplotlib.pyplot as plt
# #
# # # 绘制kd数据的准确率
# plt.plot(original_kd_index, original_kd_acc, label='original_KD', color='g', linestyle='-')
# plt.plot(kd_index, kd_acc, label='KD', color='b', linestyle=':')
#
# # 添加图例、标题和坐标轴标签
# plt.legend()
# # plt.title('Accuracy Comparison of original KD and KD')
# plt.xlabel('epoch')
# plt.ylabel('Accuracy')
#
# # 显示图表
# plt.show()



# 读取kd数据的CSV文件
kd_data = pd.read_csv('02数据增强后.csv')
kd_acc = kd_data['Train Accuracy']
kd_loss = kd_data['Train Loss']

Dynamic_T_data = pd.read_csv('动态t.csv')
Dynamic_T_acc = Dynamic_T_data['Train Accuracy']
Dynamic_T_loss = Dynamic_T_data['Train Loss']

# 如果CSV文件中包含x轴的数据（例如迭代次数），请相应地读取
# 这里假设没有x轴数据，所以我们创建一个简单的索引
kd_index = range(len(kd_acc))
Dynamic_T_index = range(len(Dynamic_T_acc))

import matplotlib.pyplot as plt
#
# # 绘制kd数据的准确率
plt.plot(kd_index, kd_acc, label='KD', color='g', linestyle='-')
plt.plot(Dynamic_T_index, Dynamic_T_acc, label='DYTKD', color='b', linestyle=':')

# 添加图例、标题和坐标轴标签
plt.legend()
# plt.title('Accuracy Comparison of original KD and KD')
plt.xlabel('epoch')
plt.ylabel('Accuracy')

# 显示图表
plt.show()
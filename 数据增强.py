from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageDraw
# import cv2
import numpy as np
import matplotlib.pyplot as plt




# 打开图像
image_path1 = r"/tmp/pycharm_project_594/dataset/PlantVillage/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/0b95527e-6cb8-4d50-95d2-1de6562b22ec___FAM_L.Blight 1702.JPG"
image_path2 = r"/tmp/pycharm_project_594/dataset/PlantVillage/Apple___Apple_scab/01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003.JPG"
image_path3 = r"/tmp/pycharm_project_594/dataset/PlantVillage/Corn_(maize)___Common_rust_/RS_Rust 1585.JPG"
image_path4 = r"/tmp/pycharm_project_594/dataset/PlantVillage/Strawberry___Leaf_scorch/0c6d5127-86d7-4956-a93e-09963cf52f7f___RS_L.Scorch 0913.JPG"
original_image1 = Image.open(image_path1)
original_image2 = Image.open(image_path2)
original_image3 = Image.open(image_path3)
original_image4 = Image.open(image_path4)

# 设置图像显示的大小
plt.figure(figsize=(15, 15))

# 显示第一个图像及其数据增强处理
plt.subplot(4, 6, 1)
plt.title('Original Image')
plt.imshow(original_image1)
plt.axis('off')

# 数据增强方法1：旋转
rotated_image1 = original_image1.rotate(45)
plt.subplot(4, 6, 2)
plt.title('Rotated Image')
plt.imshow(rotated_image1)
plt.axis('off')

""" 亮度 """
# 数据增强方法3：亮度调整
brightness_factor = 1.5  # 增加亮度的因子
enhancer = ImageEnhance.Brightness(original_image1)
brightened_image1 = enhancer.enhance(brightness_factor)
# 显示亮度调整后的图像
plt.subplot(4, 6, 3)
plt.title('Brightened Image')
plt.imshow(brightened_image1)
plt.axis('off')

# 数据增强方法8：高斯模糊
blurred_image1 = original_image1.filter(ImageFilter.GaussianBlur(radius=7))
# 显示高斯模糊后的图像
plt.subplot(4, 6, 4)
plt.title('Blurred Image')
plt.imshow(blurred_image1)
plt.axis('off')

""" 色彩增强 """
color_enhancer = ImageEnhance.Color(original_image1)
enhanced_color_image1 = color_enhancer.enhance(2.0)  # 增强色彩
# 显示色彩增强后的图像
plt.subplot(4, 6, 5)
plt.title('Enhanced Color Image')
plt.imshow(enhanced_color_image1)
plt.axis('off')

# 数据增强方法7：饱和度调整
saturation_factor = 3  # 增加饱和度的因子
enhancer = ImageEnhance.Color(original_image1)
saturated_image1 = enhancer.enhance(saturation_factor)
# 显示饱和度调整后的图像
plt.subplot(4, 6, 6)
plt.title('Saturated Image')
plt.imshow(saturated_image1)
plt.axis('off')


# 显示第二个图像及其数据增强处理
plt.subplot(4, 6, 7)
# plt.title('Original Image')
plt.imshow(original_image2)
plt.axis('off')

rotated_image2 = original_image2.rotate(45)
plt.subplot(4, 6, 8)
# plt.title('Rotated Image 1')
plt.imshow(rotated_image2)
plt.axis('off')

""" 亮度 """
# 数据增强方法3：亮度调整
brightness_factor = 1.5  # 增加亮度的因子
enhancer = ImageEnhance.Brightness(original_image2)
brightened_image2 = enhancer.enhance(brightness_factor)
# 显示亮度调整后的图像
plt.subplot(4, 6, 9)
# plt.title('Brightened Image2')
plt.imshow(brightened_image2)
plt.axis('off')

# 数据增强方法8：高斯模糊
blurred_image2 = original_image2.filter(ImageFilter.GaussianBlur(radius=7))
# 显示高斯模糊后的图像
plt.subplot(4, 6, 10)
# plt.title('Blurred Image2')
plt.imshow(blurred_image2)
plt.axis('off')

""" 色彩增强 """
color_enhancer = ImageEnhance.Color(original_image2)
enhanced_color_image2 = color_enhancer.enhance(2.0)  # 增强色彩
# 显示色彩增强后的图像
plt.subplot(4, 6, 11)
# plt.title('Enhanced Color Image2')
plt.imshow(enhanced_color_image2)
plt.axis('off')

# 数据增强方法7：饱和度调整
saturation_factor = 3  # 增加饱和度的因子
enhancer = ImageEnhance.Color(original_image2)
saturated_image2 = enhancer.enhance(saturation_factor)
# 显示饱和度调整后的图像
plt.subplot(4, 6, 12)
# plt.title('Saturated Image2')
plt.imshow(saturated_image2)
plt.axis('off')

plt.subplot(4, 6, 13)
# plt.title('Original Image 2')
plt.imshow(original_image3)
plt.axis('off')

rotated_image3 = original_image3.rotate(45)
plt.subplot(4, 6, 14)
# plt.title('Rotated Image 1')
plt.imshow(rotated_image3)
plt.axis('off')

""" 亮度 """
# 数据增强方法3：亮度调整
brightness_factor = 1.5  # 增加亮度的因子
enhancer = ImageEnhance.Brightness(original_image3)
brightened_image3 = enhancer.enhance(brightness_factor)
# 显示亮度调整后的图像
plt.subplot(4, 6, 15)
# plt.title('Brightened Image2')
plt.imshow(brightened_image3)
plt.axis('off')

# 数据增强方法8：高斯模糊
blurred_image3 = original_image3.filter(ImageFilter.GaussianBlur(radius=7))
# 显示高斯模糊后的图像
plt.subplot(4, 6, 16)
# plt.title('Blurred Image2')
plt.imshow(blurred_image3)
plt.axis('off')

""" 色彩增强 """
color_enhancer = ImageEnhance.Color(original_image3)
enhanced_color_image3 = color_enhancer.enhance(2.0)  # 增强色彩
# 显示色彩增强后的图像
plt.subplot(4, 6, 17)
# plt.title('Enhanced Color Image2')
plt.imshow(enhanced_color_image3)
plt.axis('off')

# 数据增强方法7：饱和度调整
saturation_factor = 3  # 增加饱和度的因子
enhancer = ImageEnhance.Color(original_image3)
saturated_image3 = enhancer.enhance(saturation_factor)
# 显示饱和度调整后的图像
plt.subplot(4, 6, 18)
# plt.title('Saturated Image2')
plt.imshow(saturated_image3)
plt.axis('off')


plt.subplot(4, 6, 19)
# plt.title('Original Image')
plt.imshow(original_image4)
plt.axis('off')

rotated_image4 = original_image4.rotate(45)
plt.subplot(4, 6, 20)
# plt.title('Rotated Image')
plt.imshow(rotated_image4)
plt.axis('off')

""" 亮度 """
# 数据增强方法3：亮度调整
brightness_factor = 1.5  # 增加亮度的因子
enhancer = ImageEnhance.Brightness(original_image4)
brightened_image4 = enhancer.enhance(brightness_factor)
# 显示亮度调整后的图像
plt.subplot(4, 6, 21)
# plt.title('Brightened Image')
plt.imshow(brightened_image4)
plt.axis('off')

# 数据增强方法8：高斯模糊
blurred_image4 = original_image4.filter(ImageFilter.GaussianBlur(radius=7))
# 显示高斯模糊后的图像
plt.subplot(4, 6, 22)
# plt.title('Blurred Image')
plt.imshow(blurred_image4)
plt.axis('off')

""" 色彩增强 """
color_enhancer = ImageEnhance.Color(original_image4)
enhanced_color_image4 = color_enhancer.enhance(2.0)  # 增强色彩
# 显示色彩增强后的图像
plt.subplot(4, 6, 23)
# plt.title('Enhanced Color Image')
plt.imshow(enhanced_color_image4)
plt.axis('off')

# 数据增强方法7：饱和度调整
saturation_factor = 3  # 增加饱和度的因子
enhancer = ImageEnhance.Color(original_image4)
saturated_image4 = enhancer.enhance(saturation_factor)
# 显示饱和度调整后的图像
plt.subplot(4, 6, 24)
# plt.title('Saturated Image')
plt.imshow(saturated_image4)
plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.01)
# 显示所有图像
plt.show()
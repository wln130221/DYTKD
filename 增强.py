from PIL import Image,ImageEnhance
import os
import random
import numpy as np



# 定义数据增强函数
def augment_data(image_path, save_dir, augment_factor=1):
    # 打开图片
    image = Image.open(image_path)
    # 转换为RGB模式（如果是RGBA或其他模式）
    image = image.convert('RGB')
    # 原始图片的基本名称
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    # 保存原始图片
    image.save(os.path.join(save_dir, base_name + '_original.jpg'))
    # 随机角度旋转
    angle = random.randint(-45, 45)
    rotated_image = image.rotate(angle)
    # 随机水平翻转
    # if random.random() > 0.5:
    #     rotated_image = ImageOps.mirror(rotated_image)
    #添加随机噪声，亮度增强
    # noisy_image = add_random_noise(rotated_image)
    enhanced_image = enhance_brightness_contrast(rotated_image)
    # 保存增强后的图片
    save_name = f"{base_name}_enhanced.jpg"
    enhanced_image.save(os.path.join(save_dir, save_name))

def add_random_noise(image, noise_factor=0.05):
    np_image = np.array(image)
    # 生成随机噪声
    noise = np.random.normal(scale=noise_factor, size=np_image.shape).astype(np.uint8)
    # 添加噪声到图片
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    # 转换回 PIL Image 对象
    noisy_image = Image.fromarray(noisy_image)
    return noisy_image
def enhance_brightness_contrast(image):

    # 调整亮度
    brightness_factor = 1.5  # 增加亮度的因子
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(brightness_factor)

    # 随机调整对比度
    # contrast_factor = random.uniform(*contrast_factor_range)
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(contrast_factor)

    return brightened_image
# 假设原始数据集和保存目录
original_dataset_path = "./original_Rice Leaf Disease Images/train"
save_path = "./enhance_Rice Leaf Disease Images2/train"

# 创建保存目录
os.makedirs(save_path, exist_ok=True)

# 遍历原始数据集文件夹中的图片
for class_name in os.listdir(original_dataset_path):
    class_dir = os.path.join(original_dataset_path, class_name)
    save_class_dir = os.path.join(save_path, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # 进行数据增强
        augment_data(img_path, save_class_dir)

print("数据增强完成！")



from PIL import Image, ImageEnhance
import os
import random
import numpy as np

# 定义数据增强函数
def augment_data(image_path, save_dir, augment_factor=1):
    # 打开图片
    image = Image.open(image_path)
    # 转换为RGB模式（如果是RGBA或其他模式）
    image = image.convert('RGB')
    # 原始图片的基本名称
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    # 保存原始图片
    image.save(os.path.join(save_dir, base_name + '_original.jpg'))
    # 随机角度旋转
    angle = random.randint(-45, 45)
    rotated_image = image.rotate(angle)
    # 随机水平翻转
    # if random.random() > 0.5:
    #     rotated_image = ImageOps.mirror(rotated_image)
    # 添加随机噪声，亮度增强
    # noisy_image = add_random_noise(rotated_image)
    enhanced_image = enhance_brightness_contrast(rotated_image)
    # 保存增强后的图片
    save_name = f"{base_name}_enhanced.jpg"
    enhanced_image.save(os.path.join(save_dir, save_name))

def add_random_noise(image, noise_factor=0.05):
    np_image = np.array(image)
    # 生成随机噪声
    noise = np.random.normal(scale=noise_factor, size=np_image.shape).astype(np.uint8)
    # 添加噪声到图片
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    # 转换回 PIL Image 对象
    noisy_image = Image.fromarray(noisy_image)
    return noisy_image

def enhance_brightness_contrast(image):
    # 调整亮度
    brightness_factor = 1.5  # 增加亮度的因子
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(brightness_factor)
    return brightened_image

# 假设测试集和保存目录
test_dataset_path = "./original_Rice Leaf Disease Images/test"
save_path = "./enhance_Rice Leaf Disease Images2/test"

# 创建保存目录
os.makedirs(save_path, exist_ok=True)

# 遍历测试集文件夹中的图片
for class_name in os.listdir(test_dataset_path):
    class_dir = os.path.join(test_dataset_path, class_name)
    save_class_dir = os.path.join(save_path, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # 进行数据增强
        augment_data(img_path, save_class_dir)

print("测试集数据增强完成！")

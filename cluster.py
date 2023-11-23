import os
from sklearn.cluster import KMeans
import torch
from src.preprocess import FixedHeightResize, TRANSFORMS_EVAL
from src.model import FontResNet
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resent_weight_path = './outputs/2023-10-25/23-51-22/saved_models/ResNet_weights_19.pth'
IGNORE_SIZE = 30
CLUSTER_NUM = 100


def cluster():
    # 初始化模型并加载权重
    model = FontResNet(2383, 'resnet50').to(DEVICE)
    model._load_weights(resent_weight_path)
    
    # 获取所有的图像路径
    image_dir = "test_images/text_patches/"
    all_images = os.listdir(image_dir)  # 获取目录中的所有文件名
    image_list = [os.path.join(image_dir, image_name) for image_name in all_images]
    
    embeddings = []
    active_img_name_list = []
    ignore_count = 0
    for image_path in tqdm(image_list):
        with torch.no_grad():
            img_ = Image.open(image_path)
            
            # 检查图像的高度，如果小于40像素，则跳过
            if img_.height < IGNORE_SIZE or img_.width < IGNORE_SIZE:
                ignore_count += 1
                continue
            
            img = TRANSFORMS_EVAL(img_)
            embedding = model._extract_embedding(img.unsqueeze(0).to(DEVICE))
            embeddings.append(embedding.cpu().numpy())
            active_img_name_list.append(image_path.split('/')[-1])
    embeddings_array = np.vstack(embeddings)
    # print(f'ignore {ignore_count} images.')
    print(f'active images: {len(image_list) - ignore_count}')
    print('begin cluster.')
    # 使用 KMeans 聚类
    kmeans = KMeans(n_clusters=CLUSTER_NUM).fit(embeddings_array)
    labels = kmeans.labels_
    
    # 保存每个类的图像路径
    clusters = {}
    for i in range(CLUSTER_NUM):
        clusters[f"cluster{i}"] = [active_img_name_list[j].split('/')[-1] for j, label in enumerate(labels) if label == i]

    # 保存到文件
    with open("clusters.json", "w") as f:
        json.dump(clusters, f)

def copy_cluster_util(json_path:str, source_dir:str, output_dir:str)->None:
    # 从clusters.json加载聚类数据
    with open(json_path, "r") as f:
        clusters = json.load(f)

    # 为每个聚类创建一个子目录，并复制相关的图像到子目录中
    for cluster_name, image_names in clusters.items():
        cluster_dir = os.path.join(output_dir, cluster_name)
        
        # 创建子目录（如果尚未存在）
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        # 复制每个图像到子目录中
        for image_name in image_names:
            source_path = os.path.join(source_dir, image_name)
            dest_path = os.path.join(cluster_dir, image_name)
            shutil.copy2(source_path, dest_path)
    print("Images copied to cluster directories.")

def output_freqdis():
    # 初始化模型并加载权重
    model = FontResNet(2383, 'resnet50').to(DEVICE)
    model._load_weights(resent_weight_path)
    model.eval()
    # 获取所有的图像路径
    image_dir = "test_images/text_patches/"
    all_images = os.listdir(image_dir)  # 获取目录中的所有文件名
    image_list = [os.path.join(image_dir, image_name) for image_name in all_images]
    
    font_distribution = {}  # 使用字典来计数每个类的出现次数
    ignore_count = 0
    for image_path in tqdm(image_list):
        with torch.no_grad():
            img_ = Image.open(image_path)
            
            # 检查图像的高度，如果小于40像素，则跳过
            if img_.height < IGNORE_SIZE or img_.width < IGNORE_SIZE:
                ignore_count += 1
                continue
            
            img = TRANSFORMS_EVAL(img_)
            class_id = int(model(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).cpu().numpy())
            # print(class_id)
            font_distribution[class_id] = font_distribution.get(class_id, 0) + 1  # 更新计数器

    print(f'ignore {ignore_count} images.')
    print(f'active images: {len(image_list) - ignore_count}')
    # font_distribution_ = sorted(font_distribution.items(), key=lambda x: x[0])
    # 绘制字体分布
    with open("font_distribution.json", "w") as f:
        json.dump(font_distribution, f)
    plt.figure(figsize=(15,5))
    plt.bar(font_distribution.keys(), font_distribution.values(), width=1)
    plt.xlabel("Font Class")
    plt.ylabel("Number of Images")
    plt.title("Font Distribution")
    plt.savefig("font_distribution.png")
        
if __name__ == '__main__':
    output_freqdis()
    # copy_cluster_util("clusters.json", "test_images/text_patches/", "test_images/clusters/")

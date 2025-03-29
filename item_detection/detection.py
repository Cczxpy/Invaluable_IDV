import os

import cv2
import numpy as np
import pandas as pd


def load_name_mapping(csv_path):
    """
    从CSV文件加载小图ID到名称的映射
    
    参数:
    - csv_path (str): CSV文件路径
    
    返回:
    - id_to_name (dict): ID到名称的映射字典
    """
    df = pd.read_csv(csv_path)
    id_to_name = {}
    for _, row in df.iterrows():
        id_to_name[str(row['id'])] = row['name']
    return id_to_name

def load_images(folder_path):
    """
    加载指定文件夹中的所有图像
    
    参数:
    - folder_path (str): 图像文件夹路径
    
    返回:
    - images (dict): 文件名到图像的映射字典
    """
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # 提取文件名（不含扩展名）作为键
                name = os.path.splitext(filename)[0]
                images[name] = img
    return images

def feature_matching(large_img, small_img, method='sift', threshold=0.7):
    """
    使用特征匹配找到大图中的小图位置
    
    参数:
    - large_img (numpy.ndarray): 大图像
    - small_img (numpy.ndarray): 小图像
    - method (str): 使用的特征检测方法，'sift'或'orb'
    - threshold (float): 特征匹配的阈值，越小越严格
    
    返回:
    - matched (bool): 是否找到匹配
    - box (numpy.ndarray): 匹配区域的边界框坐标，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    - score (float): 匹配的置信度分数
    """
    # 转换为灰度图
    large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # 特征检测和描述
    if method == 'sift':
        detector = cv2.SIFT_create()
    else:  # 使用ORB
        detector = cv2.ORB_create(nfeatures=2000)
    
    # 检测关键点和描述符
    kp1, des1 = detector.detectAndCompute(small_gray, None)
    kp2, des2 = detector.detectAndCompute(large_gray, None)
    
    # 如果没有足够的特征点，返回匹配失败
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return False, None, 0
    
    # 特征匹配
    if method == 'sift':
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # 应用比率测试筛选良好匹配
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
    else:  # 使用ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        # 根据距离排序筛选良好匹配
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * threshold)]
    
    # 如果良好匹配数量不足，返回匹配失败
    min_match_count = 10
    if len(good_matches) < min_match_count:
        return False, None, 0
    
    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 如果找不到有效的单应性矩阵，返回匹配失败
    if M is None:
        return False, None, 0
    
    # 计算小图在大图中的边界框
    h, w = small_gray.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    # 计算匹配得分（内点占比）
    score = mask.sum() / len(mask) if len(mask) > 0 else 0
    
    return True, dst, score

def draw_matches(large_img, small_img_name, box, name):
    """
    在大图上绘制匹配框和名称
    
    参数:
    - large_img (numpy.ndarray): 要绘制的大图像
    - small_img_name (str): 小图像名称，用于调试
    - box (numpy.ndarray): 边界框坐标
    - name (str): 要显示的名称
    
    返回:
    - img_with_box (numpy.ndarray): 带有边界框和名称的图像
    """
    img_with_box = large_img.copy()
    
    # 绘制边界框
    box = np.int32(box)
    cv2.polylines(img_with_box, [box], True, (0, 255, 0), 3)
    
    # 计算文本位置（边界框上方）
    text_x = min(box[:, 0, 0])
    text_y = min(box[:, 0, 1]) - 10
    
    # 确保文本在图像内
    text_y = max(text_y, 30)  # 至少有30像素的空间来显示文本
    
    # 绘制背景框使文本更易读
    text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img_with_box, 
                  (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), 
                  (0, 0, 0), -1)
    
    # 绘制名称文本
    cv2.putText(img_with_box, name, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return img_with_box

def process_images(input_folder, pic_folder, csv_path, output_folder, score_threshold=0.7):
    """
    处理所有图像，标记匹配结果并保存
    
    参数:
    - input_folder (str): 大图所在文件夹
    - pic_folder (str): 小图所在文件夹
    - csv_path (str): CSV文件路径
    - output_folder (str): 结果输出文件夹
    - score_threshold (float): 匹配得分阈值，默认0.7
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载ID到名称的映射
    id_to_name = load_name_mapping(csv_path)
    
    # 加载大图和小图
    large_images = load_images(input_folder)
    small_images = load_images(pic_folder)
    
    # 对每个大图进行处理
    for large_img_name, large_img in large_images.items():
        # 复制大图用于绘制结果
        result_img = large_img.copy()
        
        print(f"处理大图: {large_img_name}")
        
        # 存储匹配结果
        matches = []
        
        # 对每个小图进行匹配
        for small_img_name, small_img in small_images.items():
            # 尝试从ID转换为名称
            display_name = id_to_name.get(small_img_name, small_img_name)
            
            # 特征匹配
            matched, box, score = feature_matching(large_img, small_img)
            
            if matched and score >= score_threshold:
                print(f"  在大图 {large_img_name} 中找到匹配: {display_name}, 得分: {score:.2f}")
                matches.append((small_img_name, box, display_name, score))
        
        # 按照得分排序，通常保留得分最高的匹配
        matches.sort(key=lambda x: x[3], reverse=True)
        
        # 绘制所有匹配
        for small_img_name, box, display_name, score in matches:
            result_img = draw_matches(result_img, small_img_name, box, display_name)
        
        # 保存结果图像
        output_path = os.path.join(output_folder, f"{large_img_name}_detected.jpg")
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    # 设置文件夹路径
    input_folder = "input"  # 大图文件夹
    pic_folder = "pic"      # 小图文件夹
    csv_path = "name_id.csv"  # ID到名称映射文件
    output_folder = "output"  # 结果输出文件夹
    score_threshold = 0.7  # 匹配得分阈值
    
    # 处理图像
    process_images(input_folder, pic_folder, csv_path, output_folder, score_threshold)

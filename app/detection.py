import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import config


def load_name_mapping(csv_path):
    """
    从CSV文件加载小图ID到名称的映射

    参数:
    - csv_path (str): CSV文件路径

    返回:
    - id_to_name (dict): ID到名称的映射字典
    """
    df = pd.read_csv(csv_path, encoding='gbk')
    id_to_name = {}
    for _, row in df.iterrows():
        # 检查id是否为有效值，跳过NaN值
        if pd.notna(row["id"]):
            id_to_name[str(int(row["id"]))] = row["name"]
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
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # 提取文件名（不含扩展名）作为键
                name = os.path.splitext(filename)[0]
                images[name] = img
    return images


def feature_matching(
    large_img,
    small_img,
    kp1=None,
    des1=None,
    kp2=None,
    des2=None,
    method="sift",
    threshold=0.7,
):
    """
    使用特征匹配找到大图中的小图位置, 可选地使用缓存的特征点和描述符.

    Args:
        large_img (np.ndarray, shape=(H, W, 3)): 大图像 (BGR format).
        small_img (np.ndarray, shape=(h, w, 3)): 小图像 (BGR format).
        kp1 (list of cv2.KeyPoint, optional): 小图的预计算关键点. Defaults to None.
        des1 (np.ndarray, optional): 小图的预计算描述符. Defaults to None.
        kp2 (list of cv2.KeyPoint, optional): 大图的预计算关键点. Defaults to None.
        des2 (np.ndarray, optional): 大图的预计算描述符. Defaults to None.
        method (str): 使用的特征检测方法，'sift'或'orb'. Defaults to 'sift'.
        threshold (float): 特征匹配的阈值. Defaults to 0.7.

    Returns:
        tuple: 包含以下元素的元组:
            - matched (bool): 是否找到匹配.
            - box (np.ndarray or None): 匹配区域的边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 如果未找到匹配则为 None.
            - score (float): 匹配的置信度分数.
            - kp1 (list of cv2.KeyPoint): 计算或传入的小图关键点.
            - des1 (np.ndarray): 计算或传入的小图描述符.
            - kp2 (list of cv2.KeyPoint): 计算或传入的大图关键点.
            - des2 (np.ndarray): 计算或传入的大图描述符.
    """
    # 转换为灰度图
    large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    # 特征检测器
    if method == "sift":
        detector = cv2.SIFT_create()
    else:  # 使用ORB
        detector = cv2.ORB_create(nfeatures=2000)

    # 检测关键点和描述符 (如果未提供)
    if kp1 is None or des1 is None:
        kp1, des1 = detector.detectAndCompute(small_gray, None)
    if kp2 is None or des2 is None:
        kp2, des2 = detector.detectAndCompute(large_gray, None)

    # 如果没有足够的特征点，返回匹配失败
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return False, None, 0, kp1, des1, kp2, des2

    # 特征匹配
    if method == "sift":
        # SIFT 使用 BFMatcher 和 knnMatch + ratio test
        matcher = cv2.BFMatcher()
        # 确保 des1 和 des2 是 float32 类型
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
        matches = matcher.knnMatch(des1, des2, k=2)

        # 应用比率测试筛选良好匹配
        good_matches = []
        # 检查 matches 是否为空以及每个元素是否有足够的长度
        if matches and all(len(m) == 2 for m in matches):
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
        else:
            # 如果knnMatch返回的匹配不足，则认为没有好的匹配
            return False, None, 0, kp1, des1, kp2, des2
    else:  # 使用ORB
        # ORB 使用 BFMatcher 和 NORM_HAMMING + crossCheck
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 确保 des1 和 des2 是 uint8 类型
        if des1.dtype != np.uint8:
            des1 = des1.astype(np.uint8)
        if des2.dtype != np.uint8:
            des2 = des2.astype(np.uint8)
        matches = matcher.match(des1, des2)

        # 根据距离排序筛选良好匹配
        matches = sorted(matches, key=lambda x: x.distance)
        # ORB 的阈值通常需要调整，这里假设 threshold 适用于距离排序后的比例
        # 或者可以直接使用固定数量的匹配
        num_good_matches = max(
            10, int(len(matches) * 0.15)
        )  # 例如取最好的15%或至少10个
        good_matches = matches[:num_good_matches]

    # 如果良好匹配数量不足，返回匹配失败
    min_match_count = 10
    if len(good_matches) < min_match_count:
        return False, None, 0, kp1, des1, kp2, des2

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 如果找不到有效的单应性矩阵，返回匹配失败
    if M is None or mask is None:  # 增加对 mask 的检查
        return False, None, 0, kp1, des1, kp2, des2

    # 计算小图在大图中的边界框
    h, w = small_gray.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 计算匹配得分（内点占比）
    score = mask.sum() / len(mask) if mask is not None and len(mask) > 0 else 0

    return True, dst, score, kp1, des1, kp2, des2


def draw_matches(large_img, small_img_name, box, name):
    """
    在大图上绘制匹配框和名称 (使用Pillow解决中文乱码)

    参数:
    - large_img (np.ndarray): 要绘制的大图像 (BGR format).
    - small_img_name (str): 小图像名称，用于调试.
    - box (np.ndarray): 边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    - name (str): 要显示的名称 (可能包含中文).

    返回:
    - img_with_box (np.ndarray): 带有边界框和名称的图像 (BGR format).
    """
    img_with_box = large_img.copy()


    # 绘制边界框 (使用OpenCV)
    box_int = np.int32(box)  # Convert box coordinates to integer
    cv2.polylines(img_with_box, [box_int], True, (0, 255, 0), 3)  # Draw green polylines

    # --- 使用Pillow绘制文本 ---
    # Convert OpenCV image (BGR) to Pillow image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试查找并加载支持中文的字体
    font_size = 20  # Define font size
    # font_path = None
    # Common CJK font names/paths for different OS
    common_cjk_fonts = [
        "simhei.ttf",
        "msyh.ttf",
        "simsun.ttc",  # Windows
        "PingFang.ttc",
        "STHeiti Light.ttc",
        "Arial Unicode MS",  # macOS
        "wqy-zenhei.ttc",
        "NotoSansCJK-Regular.otf",
        "DroidSansFallbackFull.ttf",  # Linux
    ]
    font = None
    for font_name in common_cjk_fonts:
        try:
            font = ImageFont.truetype(font_name, font_size)
            # print(f"使用字体: {font_name}") # Debug: print which font is used
            break  # Stop searching once a font is found
        except IOError:
            continue  # Try next font if current one is not found

    if font is None:
        # If no suitable font is found, fallback to Pillow's default font
        print(
            "警告: 未在系统中找到合适的中文字体, 使用Pillow默认字体 (可能不支持中文)."
        )
        font = ImageFont.load_default()

    # 计算文本位置（边界框左上角上方）
    # Calculate the top-left corner of the bounding box
    min_x = int(min(box_int[:, 0, 0]))
    min_y = int(min(box_int[:, 0, 1]))

    # Use Pillow's method to get text size
    try:
        # Use getbbox for newer Pillow versions, fallback to getsize
        if hasattr(font, "getbbox"):
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            # Adjust position based on actual text bounding box offset
            text_x = min_x
            text_y = (
                min_y - text_height - 10
            )  # Position above the box with some padding
        else:
            # Fallback for older Pillow versions
            text_width, text_height = draw.textsize(name, font=font)
            text_x = min_x
            text_y = (
                min_y - text_height - 10
            )  # Position above the box with some padding
    except AttributeError:
        # Fallback if textsize/getbbox fails
        print("警告：无法获取文本尺寸，使用估计值。")
        text_width, text_height = 50, 20  # Fallback size estimate
        text_x = min_x
        text_y = min_y - text_height - 10  # Position above the box with some padding

    # 确保文本在图像顶部边界内
    text_y = max(text_y, 5)  # Keep at least 5 pixels from the top edge
    text_pos = (text_x, text_y)

    # 绘制文本背景框以提高可读性
    bg_x1 = text_pos[0] - 2
    bg_y1 = text_pos[1] - 2
    bg_x2 = text_pos[0] + text_width + 2
    bg_y2 = text_pos[1] + text_height + 2
    draw.rectangle(
        [bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0)
    )  # Black background rectangle

    # 绘制文本
    draw.text(text_pos, name, font=font, fill=(255, 255, 255))  # White text

    # Convert Pillow image (RGB) back to OpenCV image (BGR)
    img_with_box = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # --- Pillow 文本绘制结束 ---

    return img_with_box


def process_images(
    input_folder,
    pic_folder,
    csv_path,
    output_folder,
    score_threshold,
    method="sift",
    return_matches=False,
    final_score_threshold=0.7
):
    """
    处理所有图像，标记匹配结果并保存

    Args:
        input_folder (str): 大图所在文件夹.
        pic_folder (str): 小图所在文件夹.
        csv_path (str): CSV文件路径.
        output_folder (str): 结果输出文件夹.
        score_threshold (float): 匹配得分阈值. Defaults to 0.7.
        method (str): 特征匹配方法 ('sift' or 'orb'). Defaults to 'sift'.
        return_matches (bool): 是否返回匹配结果. Defaults to False.
        final_score_threshold (float): 最终RANSAC得分阈值. Defaults to 0.7.
    
    Returns:
        list or None: 如果return_matches为True，返回所有匹配的列表，否则返回None.
            每个匹配项是一个元组 (small_img_name, box, display_name, score).
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 加载ID到名称的映射
    id_to_name = load_name_mapping(csv_path)

    # 加载大图和小图
    large_images = load_images(input_folder)
    small_images = load_images(pic_folder)

    # --- 添加缓存 ---
    small_img_cache = {}  # 用于缓存小图的 kp 和 des: {small_img_name: (kp, des)}
    # ----------------

    all_matches = []  # 存储所有匹配，如果需要返回

    # 对每个大图进行处理
    for large_img_name, large_img in large_images.items():
        # 复制大图用于绘制结果
        result_img = large_img.copy()
        print(f"处理大图: {large_img_name}")

        # --- 缓存大图的 kp 和 des ---
        # 计算一次大图的特征点和描述符
        print(f"  计算大图 {large_img_name} 的特征...")
        large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
        if method == "sift":
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create(nfeatures=2000)
        kp_large, des_large = detector.detectAndCompute(large_gray, None)
        print(f"  大图 {large_img_name} 计算完成.")
        # ---------------------------

        # 存储匹配结果
        matches_found = []  # 使用 matches_found 代替 matches 避免与内部变量混淆

        # 对每个小图进行匹配
        for small_img_name, small_img in small_images.items():
            # 尝试从ID转换为名称
            # print(small_img_name)
            # print(id_to_name)
            # breakpoint()
            display_name = id_to_name.get(str(small_img_name), small_img_name)
            # print(display_name)
            # --- 使用/填充小图缓存 ---
            kp_small, des_small = None, None
            if small_img_name in small_img_cache:
                kp_small, des_small = small_img_cache[small_img_name]
            else:
                small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                if method == "sift":
                    detector_small = cv2.SIFT_create()
                else:
                    detector_small = cv2.ORB_create(nfeatures=2000)
                kp_small_calc, des_small_calc = detector_small.detectAndCompute(
                    small_gray, None
                )

                if des_small_calc is not None:  # Only cache if detection was successful
                    small_img_cache[small_img_name] = (kp_small_calc, des_small_calc)
                    kp_small, des_small = kp_small_calc, des_small_calc
                else:
                    print(f"    警告: 无法为 {small_img_name} 计算特征.")
                    continue  # Skip this small image if features can't be computed
            # -----------------------

            # 特征匹配 - 传入缓存的 kp 和 des
            matched, box, score, _, _, _, _ = feature_matching(
                large_img,
                small_img,
                kp1=kp_small,
                des1=des_small,
                kp2=kp_large,
                des2=des_large,  # Pass cached large image features
                method=method,
                threshold=score_threshold,  # Ensure method/threshold are passed
            )

            if matched:
                print(
                    f"  在大图 {large_img_name} 中找到潜在匹配: {display_name} (ID: {small_img_name}), 得分: {score:.2f}"
                )
                if score >= final_score_threshold:
                    print("    -> 满足阈值要求，添加匹配.")
                    matches_found.append((small_img_name, box, display_name, score))

        # 按照得分排序
        matches_found.sort(key=lambda x: x[3], reverse=True)
        # print(matches_found)
        # 如果需要返回匹配结果，添加到总列表中
        if return_matches:
            all_matches.extend(matches_found)

        # 绘制所有满足阈值的匹配
        for small_img_name, box, display_name, score in matches_found:
            # 确保 box 是有效的 np.ndarray
            if (
                box is not None
                and isinstance(box, np.ndarray)
                and box.shape == (4, 1, 2)
            ):
                result_img = draw_matches(result_img, small_img_name, box, display_name)
            else:
                print(
                    f"警告: 跳过绘制无效的边界框 for {display_name} in {large_img_name}."
                )

        # 保存结果图像
        output_path = os.path.join(
            output_folder, f"{large_img_name}_detected_{method}.jpg"
        )  # Add method to filename
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存至: {output_path}\n")

    # 如果需要返回匹配结果
    if return_matches:
        return all_matches
    return None


if __name__ == "__main__":
    # 设置文件夹路径
    input_folder = os.path.join(os.path.dirname(__file__), "input")  # 大图文件夹
    pic_folder = config.small_fig_path  # 小图文件夹
    csv_path = config.name_id_path  # ID到名称映射文件
    output_folder = os.path.join(os.path.dirname(__file__), "output")  # 结果输出文件夹
    score_threshold = (
        config.sift_score_threshold
    )  # SIFT得分阈值 (knn match ratio) - 调整此值
    final_score_threshold = config.final_score_threshold  # 可以增加一个最终RANSAC得分阈值，如果需要的话
    method_to_use = "sift"  # or 'orb'

    # 处理图像
    process_images(
        input_folder,
        pic_folder,
        csv_path,
        output_folder,
        score_threshold,
        method=method_to_use,
        final_score_threshold=final_score_threshold
    )  # Pass method

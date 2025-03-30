import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config


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
        id_to_name[str(row["id"])] = row["name"]
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
    threshold=0.7,  # Lowe的比率测试(ratio test)
):
    """
    使用特征匹配找到大图中的小图位置, 可选地使用缓存的特征点和描述符, 并进行几何有效性检查.

    Args:
        large_img (np.ndarray, shape=(H, W, 3)): 大图像 (BGR format).
        small_img (np.ndarray, shape=(h, w, 3)): 小图像 (BGR format).
        kp1 (list of cv2.KeyPoint, optional): 小图的预计算关键点. Defaults to None.
        des1 (np.ndarray, optional): 小图的预计算描述符. Defaults to None.
        kp2 (list of cv2.KeyPoint, optional): 大图的预计算关键点. Defaults to None.
        des2 (np.ndarray, optional): 大图的预计算描述符. Defaults to None.
        threshold (float): 特征匹配的阈值. Defaults to 0.7.


    Returns:
        tuple: 包含以下元素的元组:
            - matched (bool): 是否找到匹配.
            - box (np.ndarray or None): 匹配区域的边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 如果未找到匹配则为 None.
            - score (float): 匹配的置信度分数 (内点比例).
            - kp1 (list of cv2.KeyPoint): 计算或传入的小图关键点.
            - des1 (np.ndarray): 计算或传入的小图描述符.
            - kp2 (list of cv2.KeyPoint): 计算或传入的大图关键点.
            - des2 (np.ndarray): 计算或传入的大图描述符.
    """
    # 转换为灰度图
    large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    # 特征检测器
    detector = cv2.SIFT_create()

    # 检测关键点和描述符 (如果未提供)
    if kp1 is None or des1 is None:
        kp1, des1 = detector.detectAndCompute(small_gray, None)
    if kp2 is None or des2 is None:
        kp2, des2 = detector.detectAndCompute(large_gray, None)

    # 如果没有足够的特征点，返回匹配失败
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("特征点不足")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 特征匹配
    # SIFT 使用 BFMatcher 和 knnMatch + ratio test
    matcher = cv2.BFMatcher()
    # 确保 des1 和 des2 是 float32 类型
    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)

    # 检查描述符维度是否匹配
    if des1.shape[1] != des2.shape[1]:
        print(f"    错误: 描述符维度不匹配 - des1: {des1.shape}, des2: {des2.shape}")
        return False, None, 0, kp1, des1, kp2, des2

    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"    knnMatch 错误: {e}")
        print(f"    des1 shape: {des1.shape}, dtype: {des1.dtype}")
        print(f"    des2 shape: {des2.shape}, dtype: {des2.dtype}")
        return False, None, 0, kp1, des1, kp2, des2

    # 应用比率测试筛选良好匹配
    good_matches = []
    # 检查 matches 是否为空以及每个元素是否有足够的长度
    if matches and all(len(m) >= 2 for m in matches):  # 确保至少有2个匹配
        for m_pair in matches:
            # 检查 m_pair 的长度，以防万一只有1个匹配
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
            elif len(m_pair) == 1:
                # 如果k=2但只返回一个匹配，我们可以根据情况决定是否包含它
                # 这里我们选择忽略它，因为比率测试需要两个邻居
                pass
    else:
        # 如果knnMatch返回的匹配不足，则认为没有好的匹配
        print("    knnMatch 返回的匹配不足或格式不正确")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 如果良好匹配数量不足，返回匹配失败
    min_match_count = config.min_match_count
    if len(good_matches) < min_match_count:
        print(f"    良好匹配数量不足: {len(good_matches)} < {min_match_count}")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 如果找不到有效的单应性矩阵，返回匹配失败
    if M is None or mask is None:
        print("    找不到有效的单应性矩阵")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 计算小图在大图中的边界框
    h_small, w_small = small_gray.shape
    pts = np.float32(
        [[0, 0], [0, h_small - 1], [w_small - 1, h_small - 1], [w_small - 1, 0]]
    ).reshape(-1, 1, 2)
    try:
        dst = cv2.perspectiveTransform(pts, M)
    except cv2.error as e:
        print(f"    perspectiveTransform 错误: {e}")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 检查 dst 是否有效
    if dst is None or not isinstance(dst, np.ndarray) or dst.shape != (4, 1, 2):
        print("    dst 计算无效")  # Debug
        return False, None, 0, kp1, des1, kp2, des2

    # 添加形状检测
    if not is_rectangle_like(dst):
        print("    形状检测失败: 匹配区域不像矩形")
        return False, None, 0, kp1, des1, kp2, des2

    # 计算匹配得分（内点占比）
    score = mask.sum() / len(mask) if mask is not None and len(mask) > 0 else 0
    score = score * len(good_matches)

    print(
        f"    匹配成功! Score: {score:.2f}, 匹配点数: {len(good_matches)}/{len(matches)}, 内点数: {mask.sum()}/{len(mask)}"
    )  # Debug
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


def calculate_iou(box1, box2):
    """
    计算两个四边形的交并比(IoU)

    参数:
    - box1 (np.ndarray, shape=(4, 1, 2)): 第一个边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    - box2 (np.ndarray, shape=(4, 1, 2)): 第二个边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    返回:
    - iou (float): 两个边界框的IoU值
    """
    # 将坐标转换为cv2.contour格式
    box1_contour = box1.reshape(-1, 2).astype(np.int32)
    box2_contour = box2.reshape(-1, 2).astype(np.int32)

    # 创建二值化图像来计算交集和并集
    width = max(np.max(box1_contour[:, 0]), np.max(box2_contour[:, 0])) + 10
    height = max(np.max(box1_contour[:, 1]), np.max(box2_contour[:, 1])) + 10

    box1_mask = np.zeros((height, width), dtype=np.uint8)
    box2_mask = np.zeros((height, width), dtype=np.uint8)

    # 填充多边形
    cv2.fillPoly(box1_mask, [box1_contour], 1)
    cv2.fillPoly(box2_mask, [box2_contour], 1)

    # 计算交集和并集
    intersection = np.logical_and(box1_mask, box2_mask).sum()
    union = np.logical_or(box1_mask, box2_mask).sum()

    # 计算IoU
    if union == 0:
        return 0.0

    return intersection / union


def is_rectangle_like(
    box, aspect_ratio_threshold=3.0, angle_threshold=30.0, min_area_ratio=0.65
):
    """
    判断边界框是否接近矩形

    参数:
    - box (np.ndarray, shape=(4, 1, 2)): 边界框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    - aspect_ratio_threshold (float): 长宽比阈值，超过此值视为非方形
    - angle_threshold (float): 角度偏差阈值（度），角度偏离90度超过此值视为非方形
    - min_area_ratio (float): 最小面积比率，边界框面积与最小外接矩形面积的比率

    返回:
    - is_rect (bool): 边界框是否接近矩形
    """
    # 重塑边界框坐标
    pts = box.reshape(4, 2)

    # 计算边长
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        edge_length = np.sqrt(((p2 - p1) ** 2).sum())
        edges.append(edge_length)

    # 检查对边是否近似相等
    opposite_sides_ratio_1 = max(edges[0], edges[2]) / max(0.1, min(edges[0], edges[2]))
    opposite_sides_ratio_2 = max(edges[1], edges[3]) / max(0.1, min(edges[1], edges[3]))

    if opposite_sides_ratio_1 > 1.5 or opposite_sides_ratio_2 > 1.5:
        print("    形状检测: 对边长度不匹配")
        return False

    # 检查长宽比是否合理
    side1_avg = (edges[0] + edges[2]) / 2
    side2_avg = (edges[1] + edges[3]) / 2
    aspect_ratio = max(side1_avg, side2_avg) / max(0.1, min(side1_avg, side2_avg))

    if aspect_ratio > aspect_ratio_threshold:
        print(
            f"    形状检测: 长宽比过大 ({aspect_ratio:.2f} > {aspect_ratio_threshold})"
        )
        return False

    # 计算角度 (余弦定理)
    angles = []
    for i in range(4):
        p1 = pts[i]
        p0 = pts[(i - 1) % 4]
        p2 = pts[(i + 1) % 4]

        v1 = p0 - p1
        v2 = p2 - p1

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        # 限制范围避免浮点误差
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    # 检查角度是否接近90度
    for i, angle in enumerate(angles):
        if abs(angle - 90) > angle_threshold:
            print(
                f"    形状检测: 角度偏离90度过大 (角{i + 1}: {angle:.1f}度, 偏差: {abs(angle - 90):.1f} > {angle_threshold})"
            )
            return False

    # 计算边界框面积与最小外接矩形面积的比率
    box_area = cv2.contourArea(pts.astype(np.float32))
    rect = cv2.minAreaRect(pts.astype(np.float32))
    rect_area = rect[1][0] * rect[1][1]
    area_ratio = box_area / (rect_area + 1e-6)

    if area_ratio < min_area_ratio:
        print(f"    形状检测: 面积比率过小 ({area_ratio:.2f} < {min_area_ratio})")
        return False

    return True


def process_images(input_folder, pic_folder, csv_path, return_matches=False):
    """
    处理所有图像，标记匹配结果并保存，使用NMS去除重叠框

    Args:
        input_folder (str): 大图所在文件夹.
        pic_folder (str): 小图所在文件夹.
        csv_path (str): CSV文件路径.
        return_matches (bool): 是否返回匹配结果. Defaults to False.
        iou_threshold (float): 执行NMS时的IoU阈值. Defaults to 0.5.
        aspect_ratio_threshold (float): 形状检测的长宽比阈值. Defaults to 3.0.
        angle_threshold (float): 形状检测的角度阈值(度). Defaults to 30.0.
        min_area_ratio (float): 形状检测的面积比率阈值. Defaults to 0.65.

    Returns:
        list or None: 如果return_matches为True，返回所有匹配的列表，否则返回None.
            每个匹配项是一个元组 (small_img_name, box, display_name, score).
    """
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
        print(f"计算大图 {large_img_name} 的特征...")
        large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create()
        kp_large, des_large = detector.detectAndCompute(large_gray, None)
        print(f"大图 {large_img_name} 计算完成.")
        # ---------------------------

        # 存储匹配结果
        matches_found = []  # 使用 matches_found 代替 matches 避免与内部变量混淆

        # 对每个小图进行匹配
        for small_img_name, small_img in small_images.items():
            # 尝试从ID转换为名称
            display_name = id_to_name.get(small_img_name, small_img_name)

            # --- 使用/填充小图缓存 ---
            kp_small, des_small = None, None
            if small_img_name in small_img_cache:
                kp_small, des_small = small_img_cache[small_img_name]
                # print(f"    使用缓存特征 for {small_img_name}") # Debug
            else:
                # print(f"    计算特征 for {small_img_name}") # Debug
                small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
                detector_small = cv2.SIFT_create()
                kp_small_calc, des_small_calc = detector_small.detectAndCompute(
                    small_gray, None
                )

                if (
                    des_small_calc is not None and len(kp_small_calc) > 0
                ):  # Only cache if detection was successful and features exist
                    small_img_cache[small_img_name] = (kp_small_calc, des_small_calc)
                    kp_small, des_small = kp_small_calc, des_small_calc
                else:
                    print(f"警告: 无法为 {small_img_name} 计算或缓存特征.")
                    continue  # Skip this small image if features can't be computed
            # -----------------------

            # 检查小图特征是否有效
            if kp_small is None or des_small is None or len(kp_small) < 4:
                print(
                    f"跳过 {small_img_name}: 有效特征点不足 ({len(kp_small) if kp_small else 0})"
                )
                continue

            # 特征匹配 - 传入缓存的 kp 和 des 以及几何阈值
            print(f"尝试匹配: {display_name} (ID: {small_img_name})")  # Debug
            matched, box, score, _, _, _, _ = feature_matching(
                large_img,
                small_img,
                kp1=kp_small,
                des1=des_small,
                kp2=kp_large,
                des2=des_large,  # Pass cached large image features
                threshold=config.lowe_ratio_test_threshold,
            )

            if (
                matched and score >= config.score_threshold
            ):  # 注意：现在 matched=True 意味着通过了所有检查，包括几何检查
                print(
                    f"    在大图 {large_img_name} 中找到有效匹配: {display_name} (ID: {small_img_name}), 特征得分: {score:.2f}"
                )
                matches_found.append(
                    (small_img_name, box, display_name, score)
                )  # 添加所有通过检查的匹配

        # 按照得分排序 (仍然使用内点比例得分)
        matches_found.sort(key=lambda x: x[3], reverse=True)

        # 执行非极大值抑制(NMS)
        nms_results = []
        while matches_found:
            # 取得分最高的匹配结果
            best_match = matches_found.pop(0)
            nms_results.append(best_match)

            # 保留不与最佳匹配重叠的其他匹配
            remaining_matches = []
            best_box = best_match[1]  # 提取边界框

            for match in matches_found:
                current_box = match[1]
                # 计算IoU
                iou = calculate_iou(best_box, current_box)

                # 如果IoU小于阈值，保留这个匹配
                if iou < config.iou_threshold:
                    remaining_matches.append(match)
                else:
                    print(
                        f"    NMS: 移除与 {best_match[2]} 重叠的框 {match[2]} (IoU: {iou:.2f} > {config.iou_threshold}), (Score: {best_match[3]:.2f} > {match[3]:.2f})"
                    )

            # 更新剩余的匹配
            matches_found = remaining_matches

        # 使用NMS后的结果
        matches_found = nms_results

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
    method_to_use = config.method  # or 'orb'

    # 处理图像
    process_images(
        input_folder,
        pic_folder,
        csv_path,
        score_threshold,
    )  # Pass method

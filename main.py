import math
import os
import re
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app import config
from app.detection import draw_matches, process_images
from app.webjudger import WebJudger
import requests
import os
import subprocess
import threading
import time

# 全局变量，用于存储进程
frontend_process = None
crawler_process = None


def start_frontend():
    """启动前端展示服务"""
    global frontend_process
    
    if frontend_process and frontend_process.poll() is None:
        return "前端服务已在运行中！"
    
    try:
        # 使用配置中的脚本路径
        cmd = ["python", config.frontend_script_path]
        # 启动前端服务，使用subprocess.Popen使其在后台运行
        frontend_process = subprocess.Popen(cmd, cwd=config.PROJECT_ROOT)
        time.sleep(2)  # 等待服务启动
        
        if frontend_process.poll() is None:
            return "前端服务已成功启动！\n访问地址: http://127.0.0.1:5050"
        else:
            return f"前端服务启动失败！退出码: {frontend_process.returncode}"
    except Exception as e:
        return f"启动前端服务时出错: {str(e)}"



def start_crawler():
    """启动爬虫服务"""
    global crawler_process
    
    if crawler_process and crawler_process.poll() is None:
        return "爬虫服务已在运行中！"
    
    try:
        # 使用配置中的脚本路径
        cmd = ["python", config.crawler_script_path]
        # 启动爬虫服务，使用subprocess.Popen使其在后台运行
        crawler_process = subprocess.Popen(cmd, cwd=config.PROJECT_ROOT)
        time.sleep(2)  # 等待服务启动
        
        if crawler_process.poll() is None:
            return "爬虫服务已成功启动！\n请查看终端输出获取详细信息"
        else:
            return f"爬虫服务启动失败！退出码: {crawler_process.returncode}"
    except Exception as e:
        return f"启动爬虫服务时出错: {str(e)}"



def stop_frontend():
    """停止前端展示服务"""
    global frontend_process
    
    if frontend_process and frontend_process.poll() is None:
        frontend_process.terminate()
        frontend_process.wait()
        return "前端服务已成功停止！"
    else:
        return "前端服务未在运行中！"



def stop_crawler():
    """停止爬虫服务"""
    global crawler_process
    
    if crawler_process and crawler_process.poll() is None:
        crawler_process.terminate()
        crawler_process.wait()
        return "爬虫服务已成功停止！"
    else:
        return "爬虫服务未在运行中！"



def check_status():
    """检查服务状态"""
    frontend_status = "运行中" if (frontend_process and frontend_process.poll() is None) else "未运行"
    crawler_status = "运行中" if (crawler_process and crawler_process.poll() is None) else "未运行"
    
    return f"前端服务状态: {frontend_status}\n爬虫服务状态: {crawler_status}"

# 检查并下载未下载的图片
def check_and_download_images():
    print("正在检查并下载未下载的图片...")
    
    # 定义文件路径
    final_maker_path = config.name_id_path  # 使用配置中的路径
    pic_folder = config.small_fig_path  # 使用配置中的pic文件夹路径
    
    # 创建pic文件夹（如果不存在）
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
        print(f"创建文件夹成功: {pic_folder}")
    
    # 读取final_maker.csv文件
    df = pd.read_csv(final_maker_path, encoding='gbk')
    
    # 过滤出有id和path_to_cbg的数据
    df = df[df['id'].notna() & df['path_to_cbg'].notna()]
    
    # 检查并下载图片
    downloaded_count = 0
    for index, row in df.iterrows():
        try:
            # 获取id和图片URL
            img_id = str(int(float(row['id'])))
            img_url = row['path_to_cbg']
            
            # 构建保存路径
            save_path = os.path.join(pic_folder, f"{img_id}.png")
            
            # 检查文件是否已存在
            if os.path.exists(save_path):
                continue
            
            # 下载图片
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()  # 检查请求是否成功
            
            # 保存图片
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_count += 1
            print(f"已下载新图片: {img_id}.png")
            
        except Exception as e:
            print(f"下载图片失败: {img_id}, 错误: {e}")
    
    if downloaded_count > 0:
        print(f"图片检查完成，共下载 {downloaded_count} 张新图片")
    else:
        print("图片检查完成，所有图片已存在")

# 在启动时检查并下载图片
check_and_download_images()

# 继续加载配置和数据
csv_file = config.price_path
df = pd.read_csv(csv_file, encoding='gbk')
cards = []
# 将 DataFrame 转换为字典列表
cards = df.to_dict(orient="records")

boxes = []
txts = []
scores = []
ans = 0
total = 0
decc = 1


def cle(str_a):
    return re.sub(r'[“”‘’"\']', "", str_a)


def cmp(tex_a, tex_b):
    lst = 0
    tot = 0
    len_b = len(tex_b)
    len_a = len(tex_a)
    for ind in range(0, len_a):
        for i in range(lst, len_b):
            if tex_a[ind] == tex_b[i] and tex_a[ind] != "'" and ind == i:
                lst = i + 1
                tot += 1
    if tot >= 3:
        return True
    return False


def making_words(input_image_dir):
    global cards, boxes, txts, scores, ans, total, decc
    usd = {}
    decc = 1

    # 加载价格数据
    for card in cards:
        usd[card["name"]] = False
        if "price_new" in card:
            # 将字符串转换为浮点数，再向下取整为整数
            card["price_new"] = math.floor(float(card["price_new"]))
        else:
            card["price_new"] = 0  # 如果缺失 price_new，则默认值为 0

    for card in cards:
        if "price_old" in card:
            # 将字符串转换为浮点数，再向下取整为整数
            card["price_old"] = math.floor(float(card["price_old"]))
        else:
            card["price_old"] = 0  # 如果缺失 price_new，则默认值为 0

    # 设置文件夹路径
    # 大图文件夹
    input_folder = input_image_dir
    time_string = os.path.basename(os.path.normpath(input_image_dir))
    pic_folder = config.small_fig_path  # 小图文件夹
    csv_path = config.name_id_path  # ID到名称映射文件
    output_folder = config.item_detection_output_path  # 结果输出文件夹
    score_threshold = config.sift_score_threshold  # 匹配得分阈值

    # 创建输出文件夹（如果不存在）
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # 调用detection.py中的处理函数
    namelistnow = []
    name_price = {}
    for card in cards:
        name_price[card["name"]] = card["price_new"]

    # 处理图像
    matches_found = process_images(
        input_folder,
        pic_folder,
        csv_path,
        output_folder,
        score_threshold,
        method="sift",
        return_matches=True,
    )
    # 按照匹配得分对结果进行排序
    # matches_found 格式为 (small_img_name, box, display_name, score)
    if matches_found:
        # 按照得分（第4个元素，索引为3）降序排序
        matches_found.sort(key=lambda x: x[3], reverse=True)
        print(f"已按照匹配得分降序排序，共 {len(matches_found)} 个匹配项")
    else:
        print("未找到任何匹配项")

    # 处理匹配结果
    total = 0
    boxes = []
    txts = []
    scores = []

    for small_img_name, box, display_name, score in matches_found:
        if display_name in name_price:
            total += name_price[display_name]
            decc *= 0.97
            boxes.append(box)
            txts.append(display_name)
            scores.append(name_price[display_name])
            usd[display_name] = True
            namelistnow.append(display_name)

    # 计算折扣系数和最终价格
    if decc <= 0.3:
        decc = 0.3
    decc = math.log10(10 * decc)
    if decc <= 0.65:
        decc = 0.65
    ans = total * decc
    ans = math.floor(ans)
    decc = round(decc, 3)

    # 构建summary_text
    summary_text = (
        f"所标注皮肤总价格为: {total}\n建议乘折扣系数: {decc}\n得到基础价格: {ans}"
    )

    # 构建description，包含所有皮肤信息
    description = "当前所标注高价值皮肤有：\n"
    
    # 按照分数从高到低排序所有的txt和score
    all_items = list(zip(txts, scores))
    all_items.sort(key=lambda x: x[1], reverse=True)
    
    for i, (txt, score) in enumerate(all_items):
        description += f"{i + 1}   {txt}:    {score}\n"
    
    # 添加summary_text到description
    description += f"\n{summary_text}"

    # 读取原始图像
    image_path = os.path.join(input_image_dir, "0.jpg")
    image = cv2.imread(image_path)

    # 使用draw_matches函数绘制每个匹配项，只添加边界框和位置名称标注
    result_img = image.copy()
    for i, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        # 确保box是有效的np.ndarray
        if box is not None and isinstance(box, np.ndarray) and box.shape == (4, 1, 2):
            # 使用draw_matches函数绘制边界框和文本，只显示名称
            result_img = draw_matches(result_img, f"item_{i}", box, f"{txt}")

    # 保存只包含边界框和位置名称标注的结果图像
    output_image_path = os.path.join(
        config.result_output_path, time_string, "result.jpg"
    )
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, result_img)
    return output_image_path, description


# def qwen_words():
#     global total
#     device = "cuda" # the device to load the model onto
#     model = AutoModelForCausalLM.from_pretrained(
#         r"qwen",
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(r"qwen")

#     global cards,boxes,txts,scores
#     prompt = "帮我详细分析以下价格表，我所售卖的商品含有以下所有内容各一份，为我提供详细的售卖建议"
#     for txt, score in zip(txts, scores):
#         sentence = f"名字为{txt}的皮肤价格是{score}"
#         prompt = prompt + sentence

#     messages = [
#         {"role": "system", "content": "分析用户所提供内容，为用户提供详细售卖建议。比如可以告诉用户根据收藏人数调整价格高低，通过观察相似商品进一步确定价格，告诉用户在抽签期时多观察杜绝被贱卖。"},
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=1000000
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response


def process_image(input_image):
    # 生成描述文本
    global total, decc
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    input_image_path = os.path.join(
        config.input_image_path, f"{time_string}", "0.jpg"
    )
    input_image_dir = os.path.dirname(input_image_path)
    os.makedirs(input_image_dir, exist_ok=True)
    input_image.save(input_image_path)
    output_image_path, description = making_words(input_image_dir)
    output_image = Image.open(output_image_path)

    return description, output_image


web_checker = WebJudger


def webslayer(now_id):
    game_ordersn = web_checker.gain_id(now_id)
    ans_txt = ""
    ans_txt = web_checker.check_web_price(web_checker.check_new_web(game_ordersn))
    return ans_txt


with gr.Blocks() as demo:
    with gr.Tab("图鉴估价", id=1):
        gr.Markdown("# 藏宝阁AI价格预测（私信发图即可，主播私信回复后即可解锁发图）")

        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图片", height=800)
            image_output = gr.Image(label="价格标注", height=800)

        query_button = gr.Button("查询")

        with gr.Row():
            text_output = gr.Textbox(label="亮点标注", visible=False)

        query_button.click(
            process_image, inputs=image_input, outputs=[text_output, image_output]
        )

    with gr.Tab("链接估价", id=2):
        gr.Markdown("# 藏宝阁AI价格预测（发送藏宝阁号链接即可）")

        with gr.Row():
            with gr.Column():
                web_input_new = gr.Textbox(label="输入藏宝阁号链接")
                web_button = gr.Button("查询")
        with gr.Row():
            web_output = gr.Textbox(label="亮点标注")
        web_button.click(webslayer, inputs=web_input_new, outputs=web_output)

    with gr.Tab("服务控制", id=3):
        gr.Markdown("""
        # CBG Identity Evaluation
        提供前端展示和爬虫服务的启动控制
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 前端展示服务")
                frontend_btn = gr.Button("启动前端", variant="primary")
                frontend_stop_btn = gr.Button("停止前端")
                frontend_status = gr.Textbox(label="前端状态", placeholder="点击启动按钮启动前端服务")
            
            with gr.Column(scale=1):
                gr.Markdown("## 爬虫服务")
                crawler_btn = gr.Button("启动爬虫", variant="primary")
                crawler_stop_btn = gr.Button("停止爬虫")
                crawler_status = gr.Textbox(label="爬虫状态", placeholder="点击启动按钮启动爬虫服务")
    
        status_btn = gr.Button("检查服务状态")
        overall_status = gr.Textbox(label="整体状态", placeholder="点击检查状态按钮查看服务状态")
    
        # 设置按钮点击事件
        frontend_btn.click(fn=start_frontend, outputs=frontend_status)
        frontend_stop_btn.click(fn=stop_frontend, outputs=frontend_status)
        
        crawler_btn.click(fn=start_crawler, outputs=crawler_status)
        crawler_stop_btn.click(fn=stop_crawler, outputs=crawler_status)
        
        status_btn.click(fn=check_status, outputs=overall_status)

# 运行应用
demo.launch(server_name="localhost", server_port=7060)

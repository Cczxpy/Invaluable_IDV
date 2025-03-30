import math
import os
import re
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config
from detection import draw_matches, process_images
from webjudger import WebJudger

csv_file = config.price_path
df = pd.read_csv(csv_file)
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
    score_threshold = config.sift_score_threshold  # 匹配得分阈值

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

    # 为结果图像添加额外的文本信息
    # 不再将这些信息添加到txts和scores列表中，而是直接在图片上绘制
    # 这些信息将在后续代码中直接添加到图像上
    summary_text = (
        f"所标注皮肤总价格为: {total}\n建议乘折扣系数: {decc}\n得到基础价格: {ans}"
    )

    # 读取原始图像
    image_path = os.path.join(input_image_dir, "0.jpg")
    image = cv2.imread(image_path)

    # 使用draw_matches函数绘制每个匹配项
    result_img = image.copy()
    for i, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        # 确保box是有效的np.ndarray
        if box is not None and isinstance(box, np.ndarray) and box.shape == (4, 1, 2):
            # 使用draw_matches函数绘制边界框和文本
            result_img = draw_matches(result_img, f"item_{i}", box, f"{txt}: {score}")

    # 定义字体大小
    font_size = 30
    # 常见的中文字体列表
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

    # 查找可用的中文字体
    font = None
    for font_name in common_cjk_fonts:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue

    # 获取原始图像尺寸
    img_height, img_width = result_img.shape[:2]

    # 创建一个更宽的画布，右侧放置文本信息
    text_width = 500  # 文本区域宽度
    canvas_width = img_width + text_width
    canvas_height = img_height

    # 创建新的画布（白色背景）
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 将原始图像放在左侧
    canvas[:img_height, :img_width] = result_img

    # 将OpenCV图像转换为PIL图像以支持中文
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)

    # 按照分数从高到低排序所有的txt和score
    all_items = list(zip(txts, scores))
    all_items.sort(key=lambda x: x[1], reverse=True)

    # 添加所有文本信息
    y_pos = 70
    for i, (txt, score) in enumerate(all_items):
        text = f"{i + 1}. {txt}: {score}"
        if font:
            draw.text((img_width + 20, y_pos), text, font=font, fill=(0, 0, 0))
        y_pos += 40

        # 如果文本超出画布底部，调整字体大小或增加画布高度
        if y_pos > canvas_height - 40:
            # 这里可以选择增加画布高度
            new_canvas = np.ones((y_pos + 100, canvas_width, 3), dtype=np.uint8) * 255
            new_canvas[:canvas_height, :canvas_width] = cv2.cvtColor(
                np.array(pil_canvas), cv2.COLOR_RGB2BGR
            )
            canvas = new_canvas
            pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_canvas)

    # 最后输出 summary_text 中的文本信息
    y_pos += 40
    draw.text((img_width + 20, y_pos), summary_text, font=font, fill=(0, 0, 0))

    # 将PIL图像转换回OpenCV格式
    result_img = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)

    # 保存结果图像
    output_image_path = os.path.join(
        os.path.dirname(__file__), "output", time_string, "result.jpg"
    )
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, result_img)
    return output_image_path


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
        os.path.dirname(__file__), "input", f"{time_string}", "0.jpg"
    )
    input_image_dir = os.path.dirname(input_image_path)
    os.makedirs(input_image_dir, exist_ok=True)
    input_image.save(input_image_path)
    output_image = making_words(input_image_dir)
    # description = qwen_words()
    # if decc <= 0.7:
    #     decc = 0.7
    # decc = math.log10(10*decc)
    # if decc <=0.87:
    #     decc = 0.87
    # ans = total * decc
    # ans = math.floor(ans)
    description = (
        f"图中所标注皮肤总价格为{total}建议乘折扣系数{decc}，因此我给出基础价格：{ans}"
    )
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

# 运行应用
demo.launch(server_name="localhost", server_port=7060)

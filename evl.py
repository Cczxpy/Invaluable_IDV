import math
import os
import re

import gradio as gr
import pandas as pd
from paddleocr import PaddleOCR, draw_ocr

from webjudger import WebJudger

xlsx_file = os.path.join("database", "第五人格藏宝阁制作版.xlsx")
df = pd.read_excel(xlsx_file, engine="openpyxl")
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


def making_words():
    global cards, boxes, txts, scores, ans, total, decc
    usd = {}
    decc = 1
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        det_model_dir=os.path.join("pretrained_models", "ch_PP-OCRv4_det_infer"),
        rec_model_dir=os.path.join("pretrained_models", "ch_PP-OCRv4_rec_infer"),
        rec_char_dict_path=os.path.join("need", "ppocrv4_doc_dict.txt"),
        use_gpu=True,
        det_limit_side_len=4096,  # 控制检测模型输入图像的长边尺寸
        det_limit_type="max",  # 按最长边缩放（保持宽高比）
    )  # need to run only once to download and load model into memory
    img_path = r"read.jpg"

    # 读取图片并分割成高度为4000的块
    from PIL import Image

    image = Image.open(img_path)
    width, height = image.size

    # 计算需要分割的块数
    max_height = 4000
    num_chunks = math.ceil(height / max_height)

    ########################
    # 第一次识别
    ########################

    # 存储所有子图片的OCR结果
    all_results = []

    # 处理每个子图片
    for i in range(num_chunks):
        # 计算当前块的上下边界
        top = i * max_height
        bottom = min((i + 1) * max_height, height)

        # 裁剪当前块
        chunk = image.crop((0, top, width, bottom))
        chunk_path = f"chunk_{i}.jpg"
        chunk.save(chunk_path)

        # 对当前块进行OCR
        chunk_result = ocr.ocr(chunk_path, cls=True)

        # 调整坐标以反映在原始图像中的位置
        chunk_result = chunk_result[0]
        if chunk_result and len(chunk_result) > 0:
            for idx in range(len(chunk_result)):
                res = chunk_result[idx]
                box, (text, prob) = res
                # 调整坐标，加上当前块在原图中的y偏移
                adjusted_box = [[p[0], p[1] + top] for p in box]
                adjusted_res = [adjusted_box, (text, prob)]
                all_results.append(adjusted_res)

        # 删除临时文件
        os.remove(chunk_path)

    result = [all_results]

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

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

    # 显示结果
    from PIL import Image

    result = result[0]
    image = Image.open(img_path).convert("RGB")
    boxes = []
    txts = []
    scores = []
    total = 0
    boxes_tmp = [line[0] for line in result]
    txts_tmp = [line[1][0] for line in result]
    # scores_tmp = [line[1][1] for line in result]
    for idx, name_now in enumerate(txts_tmp):
        for card in cards:
            if usd[card["name"]]:
                continue
            if cle(card["name"]) == cle(name_now):
                total += card["price_new"]
                decc *= 0.97
                boxes.append(boxes_tmp[idx])
                txts.append(txts_tmp[idx])
                scores.append(card["price_new"])
                usd[card["name"]] = True
                break
            elif cmp(cle(card["name"]), cle(name_now)):
                total += card["price_new"]
                decc *= 0.97
                boxes.append(boxes_tmp[idx])
                txts.append(card["name"])
                scores.append(card["price_new"])
                usd[card["name"]] = True
                break

    if len(txts) >= 1:
        # --- 同步排序（按 scores 降序）---
        combined = list(zip(scores, boxes, txts))  # 组合
        combined_sorted = sorted(combined, key=lambda x: x, reverse=True)  # 排序
        scores_sorted, boxes_sorted, txts_sorted = zip(*combined_sorted)  # 解包

        # 覆盖原列表（转回列表类型）
        scores = list(scores_sorted)
        boxes = list(boxes_sorted)
        txts = list(txts_sorted)

    if decc <= 0.3:
        decc = 0.3
    decc = math.log10(10 * decc)
    if decc <= 0.65:
        decc = 0.65
    ans = total * decc
    ans = math.floor(ans)
    decc = round(decc, 3)

    txts.append("所标注皮肤总价格为")
    scores.append(total)
    txts.append("建议乘折扣系数")
    scores.append(decc)
    txts.append("得到基础价格")
    scores.append(ans)
    im_show = draw_ocr(image, boxes, txts, scores, font_path="./doc/fonts/simfang.ttf")
    total_price_1 = total
    im_show_1 = Image.fromarray(im_show)

    ########################
    # 第二次识别 , 偏移 offset = 100 再识别一次
    ########################

    # 存储所有子图片的OCR结果
    all_results = []

    # 处理每个子图片
    for i in range(num_chunks):
        # 计算当前块的上下边界
        top = i * max_height + 100
        bottom = min((i + 1) * max_height + 100, height)

        # 裁剪当前块
        chunk = image.crop((0, top, width, bottom))
        chunk_path = f"chunk_{i}.jpg"
        chunk.save(chunk_path)

        # 对当前块进行OCR
        chunk_result = ocr.ocr(chunk_path, cls=True)

        # 调整坐标以反映在原始图像中的位置
        chunk_result = chunk_result[0]
        if chunk_result and len(chunk_result) > 0:
            for idx in range(len(chunk_result)):
                res = chunk_result[idx]
                box, (text, prob) = res
                # 调整坐标，加上当前块在原图中的y偏移
                adjusted_box = [[p[0], p[1] + top] for p in box]
                adjusted_res = [adjusted_box, (text, prob)]
                all_results.append(adjusted_res)

        # 删除临时文件
        os.remove(chunk_path)

    result = [all_results]

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

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

    # 显示结果
    from PIL import Image

    result = result[0]
    image = Image.open(img_path).convert("RGB")
    boxes = []
    txts = []
    scores = []
    total = 0
    boxes_tmp = [line[0] for line in result]
    txts_tmp = [line[1][0] for line in result]
    # scores_tmp = [line[1][1] for line in result]
    for idx, name_now in enumerate(txts_tmp):
        for card in cards:
            if usd[card["name"]]:
                continue
            if cle(card["name"]) == cle(name_now):
                total += card["price_new"]
                decc *= 0.97
                boxes.append(boxes_tmp[idx])
                txts.append(txts_tmp[idx])
                scores.append(card["price_new"])
                usd[card["name"]] = True
                break
            elif cmp(cle(card["name"]), cle(name_now)):
                total += card["price_new"]
                decc *= 0.97
                boxes.append(boxes_tmp[idx])
                txts.append(card["name"])
                scores.append(card["price_new"])
                usd[card["name"]] = True
                break

    if len(txts) >= 1:
        # --- 同步排序（按 scores 降序）---
        combined = list(zip(scores, boxes, txts))  # 组合
        combined_sorted = sorted(combined, key=lambda x: x, reverse=True)  # 排序
        scores_sorted, boxes_sorted, txts_sorted = zip(*combined_sorted)  # 解包

        # 覆盖原列表（转回列表类型）
        scores = list(scores_sorted)
        boxes = list(boxes_sorted)
        txts = list(txts_sorted)

    if decc <= 0.3:
        decc = 0.3
    decc = math.log10(10 * decc)
    if decc <= 0.65:
        decc = 0.65
    ans = total * decc
    ans = math.floor(ans)
    decc = round(decc, 3)

    txts.append("所标注皮肤总价格为")
    scores.append(total)
    txts.append("建议乘折扣系数")
    scores.append(decc)
    txts.append("得到基础价格")
    scores.append(ans)
    im_show = draw_ocr(image, boxes, txts, scores, font_path="./doc/fonts/simfang.ttf")
    total_price_2 = total
    im_show_2 = Image.fromarray(im_show)

    ########################
    # 选择最高价
    ########################

    if total_price_1 > total_price_2:
        total = total_price_1
        im_show = im_show_1
    else:
        total = total_price_2
        im_show = im_show_2

    print(f"total_price_1: {total_price_1}, total_price_2: {total_price_2}")
    im_show.save("result.jpg")
    return r"result.jpg"


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
    input_image.save("read.jpg")
    output_image = making_words()
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

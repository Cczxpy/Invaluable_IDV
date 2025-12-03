from flask import Flask, render_template
import json
import math
from datetime import datetime, timedelta
import pandas as pd

# 导入配置
from app import config

app = Flask(__name__)

# 使用配置中的数据库文件
csv_file = config.name_id_path
df = pd.read_csv(csv_file, encoding='gb18030')

cards = []
# 将 DataFrame 转换为字典列表（与原逻辑一致）
cards = df.to_dict(orient='records')

now = datetime.now()
time_stage = '上旬'

if now.day <= 10 :
    time_stage = '上旬'
if now.day > 10 and now.day <=20 :
    time_stage = '中旬'
if now.day > 20 :
    time_stage = '下旬'


type_name = '联动'
if cards[0]['name'] == '三月兔':
    type_name = '限定'
else:
    type_name = '联动'
    time_stage = '整月'
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained(
#     r"C:\Users\21133\Desktop\mywebs\cbg\qwen",
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\21133\Desktop\mywebs\cbg\qwen")

# for i,item in enumerate(cards):
#     prompt = item['words']
#     if len(prompt) > 48 :
#         messages = [
#             {"role": "system", "content": "将其缩短到40字以内"},
#             {"role": "user", "content": prompt}
#         ]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         model_inputs = tokenizer([text], return_tensors="pt").to(device)

#         generated_ids = model.generate(
#             model_inputs.input_ids,
#             max_new_tokens=512
#         )
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         cards[i]['words'] = '(AI缩句)'+response
#         print(i,cards[i]['words'])


@app.route('/')
def home():
    # 读取 Excel 文件
    global cards,type_name,time_stage,now
    # json_file_path = r"C:\Users\21133\Desktop\mywebs\cbg\xianding_jsons\2025224.json"
    # with open(json_file_path, 'r', encoding='utf-8') as f:
    #     # 按行读取 JSON 文件，并将每行解析为字典
    #     cards = [json.loads(line) for line in f]
    
    for card in cards:
        if 'price_new' in card:
            # 将字符串转换为浮点数，再向下取整为整数
            card['price_new'] = math.floor(float(card['price_new']))
        else:
            card['price_new'] = 0  # 如果缺失 price_new，则默认值为 0

    for card in cards:
        if 'price_old' in card:
            # 将字符串转换为浮点数，再向下取整为整数
            card['price_old'] = math.floor(float(card['price_old']))
        else:
            card['price_old'] = 0  # 如果缺失 price_new，则默认值为 0        

    # 按照 price_old 从大到小排序并记录名次
    cards_sorted_by_old = sorted(cards, key=lambda x: x.get('price_old', 0), reverse=True)
    for index, card in enumerate(cards_sorted_by_old, start=1):
        card['rank_by_old'] = index  # 添加名次字段

    # 按照 price_new 从大到小排序并记录名次
    cards_sorted_by_new = sorted(cards, key=lambda x: x.get('price_new', 0), reverse=True)
    for index, card in enumerate(cards_sorted_by_new, start=1):
        card['rank_by_new'] = index  # 添加名次字段

    
    for card in cards:
        card['price_dif'] = card['price_new'] - card['price_old']
        card['rank_dif'] = card['rank_by_old'] - card['rank_by_new']
        card['price_check'] = abs(card['price_dif'])  
        card['rank_check'] = abs(card['rank_dif'])  

    cards = sorted(cards, key=lambda x: x.get('rank_by_new', 0), reverse=True)
    # print(cards)
    # 将 cards 数据传递给模板


    
    monthname = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    weekdays = ["无", "一", "二", "三", "四", "五", "六","日"]
    timeloop = ["影时间", "凌晨", "凌晨", "凌晨", "凌晨", "凌晨", "早晨", "早晨", "上午", "上午", "上午", "中午", "中午", "中午", "下午", "下午", "下午", "下午", "傍晚", "傍晚", "晚上", "晚上", "深夜", "深夜"]

    date_month = f"{monthname[now.month - 1]}"
    date_str = f"{monthname[now.month - 1]}·{now.day:02d}"
    time_str = f"{now.hour:02d}:{now.minute:02d}"
    weekday_str = weekdays[now.weekday()+1]
    daytime_str = timeloop[now.hour]
    
    # Moon phase calculation
    cycle_length = 29.5
    known_new_moon = datetime(2024, 3, 9, 22, 0, 0)
    days_since_known_new_moon = (now - known_new_moon).total_seconds() / 86400
    current_moon_phase_percentage = (days_since_known_new_moon % cycle_length) / cycle_length
    moondegree = 360 - int(current_moon_phase_percentage * 360)
    
    monthphase = ['NEW', 'CRESCENT', 'QUARTER', 'GIBBOUS', 'FULL', 'GIBBOUS', 'QUARTER', 'CRESCENT']
    if moondegree >= 337.5:
        phasename = monthphase[0]
    else:
        phasename = monthphase[int((moondegree + 22.5) // 45)]



    return render_template('index.html',date_str=date_str,time_stage=time_stage,type_name=type_name, cards=cards, date_month=date_month, time_str=time_str, weekday_str=weekday_str, daytime_str=daytime_str, phasename=phasename, moondegree=moondegree)

if __name__ == '__main__':
    app.run(debug=False,port=5050)

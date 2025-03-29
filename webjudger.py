from flask import Flask, request, Response, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime, timedelta
import requests
import urllib.parse
import json
import pandas as pd

class WebJudger():
    #截取账号id
    def gain_id(http_str):
        url = http_str
        start_index = len("https://id5.cbg.163.com/cgi/mweb/equip/1/")
        end_index = http_str.find("?", start_index)
        target_string = ""
        if end_index == -1:  # 如果没有找到问号，则截取到字符串末尾
            target_string = http_str[start_index:]
        else:
            target_string = http_str[start_index:end_index]
        print(target_string)
        return target_string


    # 爬取新账号
    def check_new_web(game_ordersn):
        url = f"https://cbg-other-desc.res.netease.com/id5/static/equipdesc/{game_ordersn}.json"
        headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "sec-ch-ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "referer": "https://id5.cbg.163.com/"
        }
        response = requests.get(url, headers=headers)
        return response

    #爬取旧账号
    def check_old_web(game_ordersn):
        url = "https://id5.cbg.163.com/cgi/api/get_equip_detail?client_type=h5"
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "zh-CN,zh;q=0.9",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "origin": "https://id5.cbg.163.com",
            "priority": "u=1, i",
            "referer": f"https://id5.cbg.163.com/cgi/mweb/equip/1/{game_ordersn}",
            "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-requested-with": "XMLHttpRequest"
        }
        data = {
            "serverid": "1",
            "ordersn": game_ordersn,
            "h5_device": "other",
            "app_client": "other",
            "exclude_equip_desc": "1",
            "exter": "direct"
        }
        response = requests.post(url, headers=headers, data=data)
        return response
    
    #价格分析
    def check_web_price(response):
        cloth_df = pd.read_excel('cloth.xlsx')
        results = []
        # 热门服装ID列表
        hot_cloth_ids = [7244, 12103, 13103, 7927, 7428, 12602, 3621, 7828, 14303, 13302]
        data = response.json()
        equip_desc_str = data.get('equip_desc')
        equip_desc = json.loads(equip_desc_str)
        pay_get_ingot = equip_desc.get('pay_get_ingot', 0) / 10
        hero_info = equip_desc.get('hero_info', [])
        detective_cloth_lst = equip_desc.get('detective_cloth_lst', [])
        chip = round(equip_desc.get('chip', 0) / 10000, 1)
        ingot = equip_desc.get('ingot', 0)
        cloth_lst_values = []
        for hero in hero_info:
            cloth_lst = hero.get('cloth_lst', [])
            cloth_lst_values.extend(cloth_lst)

        # 添加detective_cloth_lst的值，所有皮肤的id：cloth_lst_values
        cloth_lst_values.extend(detective_cloth_lst)

        # 匹配cloth_lst中的值与cloth.xlsx中的id
        matched_data = cloth_df[cloth_df['id'].isin(cloth_lst_values)]

        # 定义需要筛选的xianding值
        xianding_list = ["限定", "限定真理"]

        # 统计每个类别下的数量和名称
        result_xianding = {}
        for xianding in xianding_list:
            filtered_data = matched_data[matched_data['xianding'] == xianding]
            names = filtered_data['name'].tolist()
            prices = filtered_data['xdprice'].tolist()  # 获取价格列表
            prices1m = filtered_data['xdprice1m'].tolist()  # 获取一个月前的价格列表
            count = len(names)
            result_xianding[xianding] = {'count': count, 'names': names, 'prices': prices, 'prices1m': prices1m}

        limited_gold_names_with_prices = []
        limited_gold_names_with_prices1m = []
        excluded_names = {"别样魅力", "片刻闲暇", "未至之人", "工藤新一", "精致的笑容", "不屈的信仰"}
        for name, price, price1m in zip(result_xianding.get('限定', {}).get('names', []),
                                        result_xianding.get('限定', {}).get('prices', []),
                                        result_xianding.get('限定', {}).get('prices1m', [])):
            if name not in excluded_names:
                limited_gold_names_with_prices.append((name, price))
                limited_gold_names_with_prices1m.append((name, price1m))

        # 按价格排序
        limited_gold_names_with_prices.sort(key=lambda x: x[1], reverse=True)
        limited_gold_names_with_prices1m.sort(key=lambda x: x[1], reverse=True)

        # 获取最大值
        max_price = limited_gold_names_with_prices[0][1] if limited_gold_names_with_prices else None

        # 将排序后的结果转换为字符串
        sorted_limited_gold_names_with_prices = [f"{name}{int(price)}" for name, price in limited_gold_names_with_prices]

        # 处理限定真理名，加入价格并排序
        limited_truth_names_with_prices = []
        limited_truth_names_with_prices1m = []
        for name, price, price1m in zip(result_xianding.get('限定真理', {}).get('names', []),
                                        result_xianding.get('限定真理', {}).get('prices', []),
                                        result_xianding.get('限定真理', {}).get('prices1m', [])):
            limited_truth_names_with_prices.append((name, price))
            limited_truth_names_with_prices1m.append((name, price1m))

        # 按价格排序
        limited_truth_names_with_prices.sort(key=lambda x: x[1], reverse=True)
        limited_truth_names_with_prices1m.sort(key=lambda x: x[1], reverse=True)

        # 将排序后的结果转换为字符串
        sorted_limited_truth_names_with_prices = [f"{name}{int(price)}" for name, price in limited_truth_names_with_prices]

        # 定义需要筛选的quality值
        quality_list = ["紫皮", "金皮", "虚妄"]

        # 统计每个品质下的数量
        result_quality = {}
        for quality in quality_list:
            filtered_data = matched_data[matched_data['quality'] == quality]
            count = len(filtered_data)
            names = filtered_data['name'].tolist()
            result_quality[quality] = {'count': count, 'names': names}

        # 检查热门服装ID
        hot_cloth_names = []
        for cloth_id in hot_cloth_ids:
            if cloth_id in cloth_lst_values:
                cloth_name = cloth_df[cloth_df['id'] == cloth_id]['name'].values
                if cloth_name:
                    hot_cloth_names.append(cloth_name[0])

        # 计算所有限定类型的服装价格之和
        all_limited_cloth_prices = round((
                matched_data[matched_data['xianding'] == '限定']['xdprice'].sum() +
                matched_data[matched_data['xianding'] == '限定真理']['xdprice'].sum()
        ), 1)

        # 计算一个月前的限定类型的服装价格之和
        all_limited_cloth_prices1m = round((
                matched_data[matched_data['xianding'] == '限定']['xdprice1m'].sum() +
                matched_data[matched_data['xianding'] == '限定真理']['xdprice1m'].sum()
        ), 1)

        cloth_lst_str = json.dumps(cloth_lst_values)
        game_ordersn = "111"
        # 收集结果
        result = {
            '游戏订单号': game_ordersn,
            '所有限定价格': all_limited_cloth_prices,
            '所有限定价格1m': all_limited_cloth_prices1m,
            '氪金值': pay_get_ingot,
            '热门服装': '、'.join(hot_cloth_names),
            '限定金个数': result_xianding.get('限定', {}).get('count', 0),
            '限定金名': '、'.join(sorted_limited_gold_names_with_prices),
            '限定真理个数': result_xianding.get('限定真理', {}).get('count', 0),
            '限定真理名': '、'.join(sorted_limited_truth_names_with_prices),
            '春节限定个数': result_xianding.get('春节限定', {}).get('count', 0),
            '深渊限定个数': result_xianding.get('深渊限定', {}).get('count', 0),
            '商城限定个数': result_xianding.get('商城限定', {}).get('count', 0),
            '虚妄个数': result_quality.get('虚妄', {}).get('count', 0),
            '金皮个数': result_quality.get('金皮', {}).get('count', 0),
            '紫皮个数': result_quality.get('紫皮', {}).get('count', 0),
            'cbg链接': f"https://id5.cbg.163.com/cgi/mweb/equip/1/{game_ordersn}",
            '皮肤id': cloth_lst_str,
            '限定价max': max_price,
            '紫薯': chip,
            '回声': ingot
        }

        # 计算限定总数
        result['限定总数'] = (
                result['限定金个数'] +
                result['限定真理个数']
        )

        results.append(result)

        # 构建返回的纯文本结果
        output_text = (
            f"限定价: {int(result['所有限定价格'])}  氪金值: {int(result['氪金值'])}\n"
            f"限定价旧值（一个月前）: {int(result['所有限定价格1m'])}\n"
            f"紫薯: {result['紫薯']}万 回声: {result['回声']}\n"
            f"限定数: {result['限定总数']} 真理: {result['限定真理个数']} 金皮: {result['金皮个数']} 虚妄: {result['虚妄个数']}\n"
            f"限定：{result['限定金名']}\n"

        )

        # 找出涨价超过200的皮肤
        price_increase_skins = []
        for (name, price), (_, price1m) in zip(limited_gold_names_with_prices, limited_gold_names_with_prices1m):
            if price - price1m > 200:
                price_increase_skins.append(f"{name}{int(price - price1m)}")
        if result['限定真理名']:
            output_text += f"限定真理: {result['限定真理名']}\n"
        if price_increase_skins:
            output_text += f"涨价皮肤: {'、'.join(price_increase_skins)}\n"
        return output_text
        # return Response(output_text, content_type='text/plain; charset=utf-8'), 200



    
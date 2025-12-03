import os
import pandas as pd
import requests
from tqdm import tqdm

# 从config文件导入路径常量
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import config

# 定义文件路径
final_maker_path = config.name_id_path
pic_folder = config.small_fig_path

# 创建pic文件夹（如果不存在）
if not os.path.exists(pic_folder):
    os.makedirs(pic_folder)
    print(f"创建文件夹成功: {pic_folder}")

# 读取final_maker.csv文件
df = pd.read_csv(final_maker_path, encoding='gbk')

# 过滤出有id和path_to_cbg的数据
df = df[df['id'].notna() & df['path_to_cbg'].notna()]
print(f"共有 {len(df)} 张图片需要下载")

# 下载图片
success_count = 0
failed_count = 0
failed_urls = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="下载进度"):
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
        
        success_count += 1
        
    except Exception as e:
        failed_count += 1
        failed_urls.append((str(row['id']), img_url, str(e)))

# 输出下载结果
print(f"\n下载完成！")
print(f"成功下载: {success_count} 张")
print(f"下载失败: {failed_count} 张")

if failed_count > 0:
    print(f"\n失败的URL列表:")
    for img_id, img_url, error in failed_urls[:10]:  # 只显示前10个失败的URL
        print(f"ID: {img_id}, URL: {img_url}, 错误: {error}")
    
    if len(failed_urls) > 10:
        print(f"... 还有 {len(failed_urls) - 10} 个失败的URL未显示")

print(f"\n图片保存位置: {pic_folder}")

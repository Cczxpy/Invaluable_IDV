import pandas as pd
import re

# 读取CSV文件
file_path = 'final_maker.csv'
output_file_path = 'final_maker_with_id.csv'

# 使用GBK编码读取，适用于中文CSV文件
df = pd.read_csv(file_path, encoding='gbk')

# 定义函数提取all_cloth_list__and参数的值
def extract_cloth_id(url):
    if pd.isna(url):
        return None
    # 使用正则表达式提取all_cloth_list__and=后的数字
    match = re.search(r'all_cloth_list__and=([0-9]+)', str(url))
    if match:
        return match.group(1)
    else:
        return None

# 新增id列，提取all_cloth_list__and参数的值
df['id'] = df['usless'].apply(extract_cloth_id)

# 将结果保存为新文件
print("正在写入新文件...")
df.to_csv(output_file_path, index=False, encoding='gbk')

print(f"处理完成！已生成带id列的新文件：{output_file_path}")
print("您可以手动将该文件重命名为final_maker.csv")

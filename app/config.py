import os

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sift_score_threshold = 0.5
final_score_threshold = 0.7
small_fig_path = os.path.join(PROJECT_ROOT, "database", "pic")
name_id_path = os.path.join(PROJECT_ROOT, "database", "final_maker.csv")
price_path = os.path.join(
    PROJECT_ROOT, "database", "final_maker.csv"
)
cloth_path = os.path.join(PROJECT_ROOT, "database", "cloth.csv")

# 爬虫相关路径
cbg_data_path = os.path.join(PROJECT_ROOT, "cbg_data.json")

# 输出文件夹路径
item_detection_output_path = os.path.join(PROJECT_ROOT, "item_detection", "output")
result_output_path = os.path.join(PROJECT_ROOT, "output")
input_image_path = os.path.join(PROJECT_ROOT, "input")

# 前端相关路径
frontend_script_path = os.path.join(PROJECT_ROOT, "app", "utils", "start_display.py")
crawler_script_path = os.path.join(PROJECT_ROOT, "app", "utils", "start_crawler.py")

# 爬虫数据备份路径
save_dir = r"E:\视频制作\藏宝阁系列\号价数据统计\最终记录集"

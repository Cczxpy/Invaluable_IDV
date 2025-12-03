import os

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sift_score_threshold = 0.5
small_fig_path = os.path.join(PROJECT_ROOT, "database", "pic")
name_id_path = os.path.join(PROJECT_ROOT, "database", "final_maker.csv")
price_path = os.path.join(
    PROJECT_ROOT, "database", "final_maker.csv"
)
cloth_path = os.path.join(PROJECT_ROOT, "database", "cloth.csv")

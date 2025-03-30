import os

# KNN 匹配后应用比率测试筛选良好匹配
lowe_ratio_test_threshold = 0.5

# 如果良好匹配点小于这个值, 则认为不是好的匹配
min_match_count = 9

# 矩形检测
aspect_ratio_threshold = 3.0
angle_threshold = 30.0
min_area_ratio = 0.65

# 大于这个值的匹配小图才会被保留
score_threshold = 0.6

# NMS
iou_threshold = 0.4

# path
small_fig_path = os.path.join(os.path.dirname(__file__), "database", "pic")
name_id_path = os.path.join(os.path.dirname(__file__), "database", "name_id.csv")
price_path = os.path.join(
    os.path.dirname(__file__), "database", "第五人格藏宝阁制作版2025年3月30日.csv"
)
cloth_path = os.path.join(os.path.dirname(__file__), "database", "cloth.csv")

#!/usr/bin/env python3
"""
前端展示启动脚本
直接从项目根目录运行此脚本即可启动前端展示
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from display.disply import app

if __name__ == "__main__":
    print("启动前端展示服务...")
    print("访问地址: http://127.0.0.1:5050")
    print("按 Ctrl+C 停止服务")
    app.run(debug=False, port=5050)
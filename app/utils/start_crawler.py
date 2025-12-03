#!/usr/bin/env python3
"""
爬虫启动脚本
直接从项目根目录运行此脚本即可启动爬虫
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from crawler.cbg_final_changer import LoginCrawler

if __name__ == "__main__":
    print("启动爬虫服务...")
    print("请确保已安装所需依赖")
    
    crawler = LoginCrawler()
    crawler.start_browser()
    
    try:
        # 执行登录流程
        if crawler.handle_login("你的账号", "你的密码"):
            crawler.crawl_pages()
    finally:
        crawler.close()
        print("爬虫服务已停止")
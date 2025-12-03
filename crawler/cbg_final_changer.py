from playwright.sync_api import sync_playwright
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
# from htmldecode import process_row_hd, process_ver_middle
import math
from datetime import datetime
import os
import random

# 导入配置
from app import config

class LoginCrawler:

    def init_json(self):
        """初始化空JSON文件"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        # 使用配置中的路径
        self.json_file = config.cbg_data_path  # JSON文件名
        self.csv_file = config.name_id_path  # 使用配置中的final_maker.csv路径
        self.save_dir = config.save_dir  # 爬虫数据备份路径

    # 处理 row-hd 的div
    def process_row_hd(self, soup):
        target_div = None
        row_hd_divs = soup.find_all('div', class_='row row-hd clearfix')
        
        # 遍历所有符合条件的div
        for div in row_hd_divs:
            # 检查是否存在 icon-draw
            if not div.find('i', class_='icon icon-draw'):
                if not div.find('i', class_='icon icon-protection'):
                    target_div = div
                    break
        
        # 如果所有div都有icon则取最后一个
        if not target_div and row_hd_divs:
            target_div = row_hd_divs[-1]
        
        # 提取price内容
        if target_div:
            price_div = target_div.find('div', class_='price')
            return price_div.text.strip() if price_div else None
        return None

    # 处理 ver-middle 的div
    def process_ver_middle(self, soup):
        ver_div = soup.find('span', class_='ver-middle')
        return ver_div.text.strip() if ver_div else None
    
    def start_browser(self):
        """启动浏览器并初始化配置"""
        # self.init_json()
        self.playwright = sync_playwright().start()
        # 使用有头模式方便调试（正式运行可改为 headless=True）
        self.browser = self.playwright.chromium.launch(headless=False)
        # 创建上下文保存登录状态
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        self.page = self.context.new_page()

    def save_to_json(self, data):
        """追加数据到JSON文件"""
        soup = BeautifulSoup(data, 'lxml')  # 确保已安装 lxml
        pretty_html = soup.prettify()

        # 保存到本地文件
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_file = os.path.join(project_root, "output.html")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(pretty_html)

        print(f"HTML内容已保存至 {output_file}")


    def handle_login(self, username, password):
        """处理登录流程"""
        try:
            # 访问登录页面
            self.page.goto('https://id5.cbg.163.com/cgi/mweb/show_login')
            
            # 等待页面加载
        #     self.page.wait_for_selector('input[name="username"]', state="visible")
            
        #     # 输入账号密码
        #     self.page.fill('input[name="username"]', username)
        #     self.page.fill('input[name="password"]', password)
            
        #     # 点击登录按钮
        #     with self.page.expect_navigation():
        #         self.page.click('button:has-text("登录")')
            
        #     print("已提交账号密码，等待短信验证码...")
            
        #     # 等待验证码输入框出现
        #     self.page.wait_for_selector('input[placeholder="短信验证码"]', state="visible", timeout=60000)
            
        #     # 暂停等待人工输入验证码
        #     print("请查看手机短信并输入验证码（60秒超时）")
        #     self.page.pause()  # 在此处手动输入验证码
            
        #     # 提交验证码（根据实际页面结构调整选择器）
        #     self.page.click('button:has-text("确认")')
            
        #     # 验证登录是否成功
        #     time.sleep(3)  # 等待页面跳转
        #     if "login" not in self.page.url.lower():
        #         print("登录成功！当前URL:", self.page.url)
        #         return True
        #     else:
        #         print("登录失败，请检查流程")
        #         return False
            time.sleep(30)
            return True
        except Exception as e:
            print("登录流程异常:", str(e))
            return False

    def crawl_pages(self):
        """登录后爬取多个页面"""
        df = pd.read_csv(self.csv_file, encoding='gb18030')
        cards = []
        # 将 DataFrame 转换为字典列表（与原逻辑一致）
        cards = df.to_dict(orient='records')

        for card in cards:
            try:
                self.page.goto(card['usless'])

                delay = random.uniform(4, 8)
                time.sleep(delay)
                
                print(f"正在爬取: {card['name']}")
                
                # 示例：获取页面标题
                title = self.page.title()
                print(f"页面标题: {title}")
                
                # 示例：保存页面内容
                content = self.page.content()
                soup = BeautifulSoup(content, 'lxml')
                
                card['price_old'] = card['price_new']
                price = self.process_row_hd(soup)
                numeric_str = price.replace("¥", "")
                numeric_str = numeric_str.replace(",", "")

                price_float = float(numeric_str)
                price_floor = math.floor(price_float)
                result = int(price_floor)

                card['price_new'] = result
                card['words'] = self.process_ver_middle(soup)
                if card['words']:
                    print(f"新价格：{card['price_new']}")
                    print(f"新评论：{card['words']}")
                else:
                    card['words'] = card['name']
                    print(f"新价格：{card['price_new']}")
                    print(f"新评论：{card['words']}")
                # print(content)
                # self.save_to_json(content)
                # 这里可以添加解析逻辑...
                
            except Exception as e:
                print(f"爬取 {card['name']} 失败:", str(e))
        updated_df = pd.DataFrame(cards)

        # 保存回原文件（注意参数设置）
        updated_df.to_csv(
            self.csv_file,           # 保持路径一致
            encoding='gb18030', # 保持编码一致
            index=False,        # 不保存索引列
            mode='w'            # 覆盖模式（默认）
        )
        print(f"数据已成功保存至 {self.csv_file}")

        # 存储数据备份
        # 生成日期文件名
        today = datetime.now().strftime("%Y-%m-%d")  # 格式示例：2023-10-05
        output_path = os.path.join(self.save_dir, f"{today}.csv")

        # 保存CSV文件（保持编码一致）
        df.to_csv(
            output_path,
            encoding='gb18030',
            index=False,
            errors='replace'  # 处理特殊字符问题
        )

        print(f"文件已成功保存至：{output_path}")

    def close(self):
        """关闭资源"""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

if __name__ == "__main__":
    crawler = LoginCrawler()
    crawler.start_browser()
    
    try:
        # 执行登录流程
        if crawler.handle_login("你的账号", "你的密码"):
            # 定义要爬取的URL列表
            # target_urls = [
            #     "https://id5.cbg.163.com/cgi/mweb/pl?order_by=price%20ASC&search_type=role&all_cloth_list__and=7233&refer_sn=0195DC72-AB52-CBD2-F5F3-454DB1E219FB&conds_labels=%5B%22%E6%BB%A1%E8%B6%B3%E5%85%A8%E9%83%A8%3A%E8%B0%83%E9%A6%99%E5%B8%88-%E6%91%A9%E6%B6%85%E8%8E%AB%E7%BB%AA%E6%B6%85%E4%B9%8B%E6%A2%A6%22%5D",
            #     "https://id5.cbg.163.com/cgi/mweb/pl?view_loc=search_cond&search_type=role&tfid=f_kingkong&all_cloth_list__and=11020&refer_sn=019600B4-1112-0649-1D21-2188497EC3E5&conds_labels=%5B%22%E6%BB%A1%E8%B6%B3%E5%85%A8%E9%83%A8%3A%E2%80%9C%E5%BF%83%E7%90%86%E5%AD%A6%E5%AE%B6%E2%80%9D-%E6%81%92%E6%A2%A6%22%5D&order_by=price%20ASC"
            # ]
            crawler.crawl_pages()
    finally:
        crawler.close()

        
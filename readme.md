# 藏宝阁AI价格预测系统

## 📖 项目简介

藏宝阁AI价格预测系统是一个基于计算机视觉和爬虫技术的综合性工具，主要用于第五人格游戏账号的自动估价和皮肤分析。该系统由B站UP主Cc专心破译设计，并与多位志同道合的开发者共同完成后续开发。

## 🎯 功能特点

### 核心功能
- **图鉴估价**：上传游戏截图，自动识别和标注皮肤，并计算账号价格
- **链接估价**：输入藏宝阁链接，获取账号详细信息和价格评估
- **服务控制**：管理前端展示服务和爬虫服务

### 辅助功能
- **数据爬虫**：自动从藏宝阁网站爬取最新的皮肤价格数据
- **前端展示**：可视化展示皮肤价格数据和趋势
- **自动图片下载**：自动下载缺失的皮肤图片
- **数据备份**：定期备份爬取的数据

## 📁 项目架构

```
Invaluable_IDV/
├── app/                  # 核心应用代码
│   ├── utils/            # 工具脚本
│   │   ├── start_crawler.py       # 爬虫启动脚本
│   │   └── start_display.py       # 前端展示启动脚本
│   ├── __init__.py
│   ├── config.py         # 配置文件
│   ├── detection.py      # 图片检测和匹配
│   ├── price_maker.py    # 价格计算
│   └── webjudger.py      # 网页链接处理
├── crawler/              # 爬虫模块
│   ├── __init__.py
│   └── cbg_final_changer.py       # 藏宝阁爬虫
├── database/             # 数据库文件
│   ├── pic/              # 皮肤图片
│   ├── final_maker.csv   # 核心数据文件
│   └── cloth.csv         # 服装数据
├── display/              # 前端展示模块
│   ├── static/           # 静态资源
│   ├── templates/        # HTML模板
│   ├── __init__.py
│   └── disply.py         # Flask应用
├── main.py               # 主程序入口
├── requirements.txt      # 依赖列表
└── readme.md             # 项目说明文档
```

## 🚀 快速开始

### 环境搭建

1. **创建并激活虚拟环境**
   ```bash
   git clone https://github.com/Cczxpy/Invaluable_IDV.git
   cd Invaluable_IDV
   conda create -n cbg python=3.11
   conda activate cbg
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 使用方法

1. **配置数据库**
   - 确保 `database` 文件夹中存在 `final_maker.csv` 文件
   - 可按照现有格式自行填写需要的皮肤数据
   - 系统会自动下载缺失的皮肤图片

2. **启动主程序**
   ```bash
   python main.py
   ```

3. **访问网页界面**
   - 打开浏览器，访问 `http://localhost:7060`
   - 选择相应的功能模块进行操作

## 📊 功能模块详解

### 1. 图鉴估价
- 上传游戏账号截图
- 系统自动识别截图中的皮肤
- 标注皮肤位置并计算总价格
- 生成价格报告和标注图片

### 2. 链接估价
- 输入第五人格藏宝阁链接
- 系统自动解析链接并获取账号信息
- 生成详细的价格评估报告

### 3. 服务控制
- **前端展示服务**：启动/停止前端数据展示页面
  - 访问地址：`http://127.0.0.1:5050`
- **爬虫服务**：启动/停止藏宝阁数据爬虫
  - 自动更新皮肤价格数据
  - 定期备份数据到指定目录

## ⚙️ 配置说明

### 主要配置项 (`app/config.py`)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `PROJECT_ROOT` | 项目根目录 | 自动计算 |
| `sift_score_threshold` | SIFT匹配得分阈值 | 0.5 |
| `small_fig_path` | 皮肤图片路径 | `database/pic` |
| `name_id_path` | 核心数据文件路径 | `database/final_maker.csv` |
| `price_path` | 价格数据文件路径 | `database/final_maker.csv` |
| `cloth_path` | 服装数据文件路径 | `database/cloth.csv` |
| `cbg_data_path` | 爬虫数据文件路径 | `cbg_data.json` |
| `save_dir` | 数据备份目录 | `E:\视频制作\藏宝阁系列\号价数据统计\最终记录集` |

### 自定义配置

1. 编辑 `app/config.py` 文件
2. 修改相应的配置项
3. 保存后重启程序即可生效

## 📝 数据说明

### final_maker.csv 格式

| 字段名 | 说明 |
|--------|------|
| `id` | 皮肤ID |
| `name` | 皮肤名称 |
| `price_new` | 最新价格 |
| `price_old` | 旧价格 |
| `path_to_cbg` | 皮肤图片URL |
| `usless` | 藏宝阁链接 |
| `words` | 皮肤描述 |

## 📁 文件结构说明

- **input/**：存储上传的原始图片，按时间戳分类
- **output/**：存储处理后的结果图片和报告
- **item_detection/output/**：存储物品检测的中间结果
- **cbg_data.json**：爬虫数据临时存储文件

## 🔧 二次开发

### 开发规范
- 所有文件路径均通过 `config.py` 配置，便于统一管理
- 新增功能模块应放在合适的目录下
- 遵循现有代码风格和命名规范

### 贡献方式
1. Fork 本仓库
2. 创建功能分支
3. 提交代码
4. 创建 Pull Request

### 联系方式
- B站：Cc专心破译
- 微信：itctlsssr

## 📄 许可证

本项目仅供学习交流使用，禁止商用，包括但不限于售卖、盈利性估价、直播等。

一切数据源于网易藏宝阁，一切图片源于第五人格wiki。如有侵权，请联系作者整改或删除！

## 🙏 致谢

感谢所有为项目做出贡献的开发者们，以及社区的支持和反馈！

## 📞 联系方式

- 项目地址：https://github.com/Cczxpy/Invaluable_IDV/tree/main
- 作者微信：itctlsssr
- B站主页：https://space.bilibili.com/438331902?spm_id_from=333.788.0.0

---

**更新日志**：
- 2025-03-30：初始版本发布
- 2025-12-03：增加链接估价功能
- 2024-12-03：整合爬虫和前端展示
- 2024-12-03：优化配置管理，统一文件路径

**最后更新**：2025-12-03



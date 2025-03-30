
# 项目使用须知

该项目由b站up主Cc专心破译设计，并与多为志同道合的大佬完成后续开发，代码仅供学习交流使用，禁止商用，包括但不限于售卖、盈利性估价、直播等。

项目基于本地数据库制作，已在database中设置一个例子，由于价格具有时效性，所以不要盲目直接使用该项目进行估价。

启动项目后会生成一个网页，可以拖入图片进入该网页实现自动估价与标注

# 临时环境搭建教程

## 创建虚拟环境并安装依赖

```
conda create -n cbg python=3.11
conda activate cbg
pip install -r requirements.txt
```

## 安装 paddleocr：

https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html

## qwen使用：

本项目暂时去除了AI总结模块，可以不安装transformers和pytorch。后续可能会对此进行完善，pytorch的cuda版本必须和paddlepaddle的cuda版本一致。

## paadleocr使用：

已经在need文件夹中提供了服务端和本地端的v4模型，可根据部署需要切换使用

## 项目运行：

```
python evl.py
```

## 二次开发：

如果您对这个项目感兴趣，欢迎一起完善本项目。我的微信号是
```
itctlsssr
```
欢迎添加我的微信，并成为开发团队的一员。

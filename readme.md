# Invaluable_IDV: 第五人格估价器

一个基于账号图鉴或者网易大神外部链接的第五人格估价器，可以估计账号内所有物品的大致价值。

## 项目声明

本项目由 `b站up主Cc专心破译设计` 设计，并与多位志同道合的大佬完成后续开发，代码仅供学习交流使用，禁止商用，包括但不限于售卖、盈利性估价、直播等。

项目基于本地数据库制作，所有使用的数据均保存在 `database` 文件夹中，由于价格具有时效性，所以不要盲目直接使用该项目进行估价。

Developer:

1. [@Cc专心破译](https://space.bilibili.com/438331902)
2. [EmptyBlue](www.lyt0112.com)

## 使用说明

### 前置依赖

1. 安装 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) 来管理虚拟环境, 一个 [Miniconda 安装教程](https://www.cnblogs.com/jijunhao/p/17235904.html)

2. 然后下载该仓库或 `clone`:

    ```bash
    git clone git@github.com:Cczxpy/Invaluable_IDV.git
    ```
    > [!IMPORTANT]
    > 请不要把仓库放到有中文的目录下, 可能会导致文件读取失败, 比如 `C:\Users\小绵羊\Desktop\Invaluable_IDV` 就是含有中文的目录.

3. 进入仓库目录:

    ```bash
    cd Invaluable_IDV
    ```

### 创建虚拟环境并安装依赖

```
conda create -n cbg python=3.11
conda activate cbg
pip install -r requirements.txt
```

### 运行

运行 `main.py` 文件, 即可开始估价.

```
python main.py
```

开启网页客户端后, 上传的图片会保存在 `input/<时间戳>` 文件夹中, 经过模式匹配后保存在 `output/<同一个时间戳>` 文件夹下.

## 账号图鉴特征提取技术

> [!NOTE]
> 为什么不使用神经网络？
> 
> 1. 本项目不要求图像识别具有良好的泛化性, 只需要模版匹配即可
> 2. 神经网络的训练需要大量数据, 而本项目的数据量较小, 容易造成过拟合

> [!NOTE]
> 经过测试, 使用 VLM 比如 `GPT-4o`, `Claude 3.7 Sonnet`, `Gemini 2.5 Pro` 等大模型 zero-shot 识别账号图鉴中有哪些皮肤效果对图片的清晰度和大小都比当前方法更加鲁棒, 但是囿于 VLM API 的使用成本, 暂时还没有使用, 但是如果后期该项目有收入, 可以考虑部署到网站并且使用商业 VLM 来识别.

本项目主要采用基于特征点的图像匹配技术来识别游戏截图中的物品图标。具体步骤如下：

1.  **特征提取与匹配**: 使用 SIFT 算法提取模板图像（数据库中的小图）和待检测图像（用户上传的截图）的特征点及其描述符。然后，使用 K 最近邻 (KNN) 算法在特征空间中寻找匹配对。
2.  **比率测试 (Lowe's Ratio Test)**: 为了筛选出更可靠的匹配点对，应用了 Lowe's 比率测试。通过比较最近邻和次近邻的距离比值，可以有效剔除模糊匹配。`lowe_ratio_test_threshold` 参数用于控制筛选的严格程度，因为我们是严格的匹配已有的图片，所以我们采用了较小的值，意味着只有非常相似的特征点才会被保留，产生更少但更可靠的匹配。
3.  **几何验证与评分**:
    *   **单应性矩阵 (Homography)**: 对通过比率测试的匹配点对，使用 RANSAC 算法估计一个单应性矩阵，该矩阵描述了模板图像到待检测图像的透视变换。
    *   **内点 (Inliers)**: 在 RANSAC 过程中，能够较好地符合估计出的单应性变换的匹配点被称为内点。内点的数量或比例反映了匹配在几何上的一致性。
    *   **综合评分 (Score Function)**: 为了更准确地评估匹配质量，我们设计了一个综合评分函数。该函数将通过比率测试的 KNN 匹配的数量与单应性变换的内点比例相乘。这样做的好处是同时考虑了特征空间的相似性（匹配点数）和几何空间的一致性（内点比例），从而得到更鲁棒的匹配评分。
4.  **矩形区域检测与过滤**: 在找到潜在匹配区域后，会检测该区域是否近似为矩形。如果检测到的边界框形状与矩形差异过大，则认为该匹配无效并将其删除，以排除错误的或变形严重的匹配结果。
5.  **非极大值抑制 (NMS)**: 由于模板图像可能在待检测图像中出现多次，或者多个不同的模板图像匹配到同一区域，可能会产生重叠的检测框 (Bounding Box)。为了解决这个问题，我们计算重叠检测框之间的交并比 (Intersection over Union, IoU)，并应用非极大值抑制 (NMS) 算法。NMS 会保留得分最高的检测框，并抑制掉与其重叠度较高且得分较低的其他检测框，确保每个物品只被识别一次，且保留的是最佳匹配结果。

## 二次开发

如果您对这个项目感兴趣，欢迎一起完善本项目。我的微信号是 `itctlsssr` 欢迎添加我的微信，并成为开发团队的一员。

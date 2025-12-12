# 鼓膜形变检测系统 (Tympanic Membrane Deformation Detection)

基于 CoTracker 点跟踪的耳内镜视频鼓膜形变自动检测系统。

## 📌 项目背景

当患者做 Valsalva 动作时，鼓膜会发生微小的膨隆/凹陷形变。本系统通过分析耳内镜视频中的点运动模式，自动检测和分类鼓膜的形变状态：

- **Static (静止)**: 鼓膜无形变
- **Deforming (形变中)**: 鼓膜正在发生形变
- **Peak (峰值)**: 形变达到最大并维持

## 🏗 系统架构

```
输入视频帧 → CoTracker跟踪 → 特征提取 → 状态分类 → 输出结果
     ↑              ↓              ↓           ↓
   掩码        点轨迹 [T,N,2]   多种特征    时间状态序列
```

## 📁 代码结构

```
tympanic_detection/
├── __init__.py              # 包初始化
├── preprocessing.py         # 数据加载和网格采样
├── tracking.py             # CoTracker 跟踪封装
├── quality_control.py      # 跟踪质量控制
├── homography_analysis.py  # 透视模型分析（旧方案）
├── feature_extraction.py   # 特征提取
├── classification.py       # 状态分类器
├── postprocessing.py       # 后处理
├── pipeline.py             # 完整检测流程
├── visualization.py        # 可视化工具
└── tests/
    ├── point_distance_analysis.py    # ⭐ 点间距分析（推荐）
    ├── clustering_classification.py  # 无监督聚类
    ├── supervised_classification.py  # 监督学习分类
    └── timeseries_models.py          # 时序模型对比
```

## 🔬 核心算法

### 方案对比

| 方案 | 文件 | 原理 | 适用场景 |
|:-----|:-----|:-----|:---------|
| **透视模型残差** | `homography_analysis.py` | 用单应性补偿相机运动，计算残差 | 需要精确补偿 |
| **点间距变化** ⭐ | `point_distance_analysis.py` | 直接测量点间距变化，天然抗相机运动 | 推荐使用 |

### 特征提取

从点轨迹中提取 3 类特征：

```python
# 1. 点间距比例 (距离变化)
distance_ratio = neighbor_distance_t / neighbor_distance_0
# > 1.0 = 膨胀, < 1.0 = 收缩

# 2. 位移场散度 (辐射模式)
divergence = ∂dx/∂x + ∂dy/∂y
# > 0 = 向外膨胀, < 0 = 向内收缩

# 3. 三角网格面积比例
area_ratio = triangle_area_t / triangle_area_0
# > 1.0 = 膨胀
```

### 分类方法

| 方法 | 是否需要标注 | 说明 |
|:-----|:-------------|:-----|
| 变化点检测 | ❌ | 自动找信号突变点 |
| 聚类 (K-Means/GMM) | ❌ | 无监督聚类 |
| 监督学习 (RF/SVM/MLP) | ✅ | 逐帧分类 |
| 时序模型 (LSTM/Transformer) | ✅ | 考虑时序上下文 |

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install opencv-python numpy scipy matplotlib scikit-learn
pip install ruptures  # 可选：变化点检测

# 安装本包
cd co-tracker
pip install -e .
```

### 数据格式

```
clip_0001/
├── 000001.jpg          # 视频帧
├── 000002.jpg
├── ...
└── maskB/
    ├── maskB_000001.png  # 鼓膜掩码
    ├── maskB_000002.png
    └── ...
```

### 方式1: 点间距分析（无需标注）

```bash
cd tympanic_detection/tests

python point_distance_analysis.py \
    --clip /path/to/clip_0001 \
    --output ./results \
    --radius 2
```

输出：
- 特征时间曲线
- 变化点检测
- 状态分类

### 方式2: 监督学习（需要标注）

**标注格式** (`labels.txt`):
```
clip_0001/000001 0
clip_0001/000002 0
clip_0001/000013 1    # 1=正在变化
clip_0001/000014 1
clip_0001/000023 0    # 0=无变化
```

**训练模型**:
```bash
python supervised_classification.py train \
    --clips /home/lzq/数据准备/randomforest_data_TM/train/ \
    --labels /home/lzq/数据准备/randomforest_data_TM/train/labels.txt \
    --val_clips /home/lzq/数据准备/randomforest_data_TM/val/ \
    --val_labels /home/lzq/数据准备/randomforest_data_TM/val/labels.txt \
    --output ./trained_model \
    --model rf  # rf, svm, mlp

# 预测新视频
python supervised_classification.py predict \
    --clip /path/to/new_clip \
    --model_path ./trained_model
```

### 方式3: 时序模型对比

```bash
python timeseries_models.py \
    --clips /home/lzq/数据准备/randomforest_data_TM/train/ \
    --labels /home/lzq/数据准备/randomforest_data_TM/train/labels.txt \
    --val_clips /home/lzq/数据准备/randomforest_data_TM/val/ \
    --val_labels /home/lzq/数据准备/randomforest_data_TM/val/labels.txt \
    --output ./model_comparison \
    --models hmm cnn lstm transformer
```

## 📊 输出说明

### 特征含义

| 特征 | 静止时 | 形变中 | 峰值时 |
|:-----|:-------|:-------|:-------|
| `median_ratio` | ≈ 1.0 | 上升 | 稳定 > 1.0 |
| `divergence` | ≈ 0 | > 0 | ≈ 0 |
| `area_ratio` | ≈ 1.0 | 上升 | 稳定 > 1.0 |

### 可视化

- **时间曲线图**: 各特征随时间变化
- **热力图**: 空间分布
- **状态序列**: 分类结果

## ⚙️ 参数调优

### 网格采样

```python
--grid_size 20  # 20x20 = 400个跟踪点
--radius 2      # 邻域半径（2=24个邻居）
```

### 分类阈值

在 `point_distance_analysis.py` 中的变化点检测：
```python
n_expected_changes=2  # 预期2个变化点（静止→形变→峰值）
```

## 📝 常见问题

**Q: 为什么用点间距而不是透视模型？**

透视模型需要精确估计相机运动，容易受到跟踪误差影响。点间距是相对测量，对相机运动具有天然不变性。

**Q: 需要多少标注数据？**

- 无监督方法: 0
- 监督学习 (RF/SVM): 50-100 个视频
- 时序模型 (LSTM): 100-200 个视频

**Q: GPU 是必需的吗？**

CoTracker 跟踪建议使用 GPU。分类部分可以在 CPU 上运行。

## 📄 License

MIT License

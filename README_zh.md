# Axplorer - AI 驱动的数学问题探索器

<div align="center">

**使用深度学习和经典搜索算法发现数学问题的最优解**

[English](README.md) | **简体中文**

</div>

---

## 📖 简介

Axplorer（原 PatternBoost）是一个强大的框架，结合了**Transformer 模型**和**经典搜索算法**，用于解决组合优化和离散数学问题。它通过迭代学习高质量解的模式，并生成新的候选解，已成功应用于多个前沿数学研究。

### 🔬 核心优势

- ✨ **自动化发现**: 无需人工特征工程，自动学习解空间的结构模式
- 🚀 **高效搜索**: 结合神经网络指导的采样和局部搜索
- 📊 **可解释性**: 分析学到的模式以获得数学洞察
- 🎯 **灵活扩展**: 轻松适配新的数学问题

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 使用 micromamba 创建环境
micromamba env create -f environment.yml

# 激活环境
conda activate env_axplorer
```

### 2. 训练第一个模型

最简单的例子 - 训练一个无 4-环图的最大边数问题（Turan 问题）：

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --encoding_tokens single_integer \
    --max_len 100 \
    --temperature '0.6' \
    --inc_temp '0.1'
```

**参数说明**:
- `--env_name`: 数学问题名称（见下方[可用环境](#可用环境)列表）
- `--exp_name`: 实验名称（自定义）
- `--N`: 问题规模（如图的顶点数）
- `--temperature`: 采样温度（控制多样性）

### 3. 可视化训练过程

启动 Web UI 仪表盘实时监控训练：

```bash
python ui_dashboard.py
```

然后在浏览器打开 http://localhost:7860

---

## 📋 目录

- [核心功能](#核心功能)
- [安装指南](#安装指南)
- [训练模型](#训练模型)
- [Web UI 仪表盘](#web-ui-仪表盘)
- [可用环境](#可用环境)
- [自定义问题](#自定义问题)
- [参数详解](#参数详解)
- [常见问题](#常见问题)

---

## 🎯 核心功能

### 迭代优化流程

```
                      ┌─────────────────────────────────────────────────────┐
                      │                   每个训练周期                      │
                      │                                                     │
┌──────────┐          │   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│ 生成初始 │          │   │  训练    │    │  采样    │    │  局部    │      │
│   数据   │─────────────►│  模型    │───►│  模型    │───►│  搜索    │      │
└──────────┘          │   └──────────┘    └──────────┘    └──────────┘      │
                      │         ▲                               │           │
                      │         │         ┌──────────┐          │           │
                      │         └─────────│  选择    │◄─────────┘           │
                      │                   │  最优    │                      │
                      │                   └──────────┘                      │
                      └─────────────────────────────────────────────────────┘
```

### 工作流程详解

1. **初始数据生成** (仅第一个周期)
   - 使用贪心构造生成 `gensize` 个随机有效样本
   - 根据优化目标评分
   - 保留前 `pop_size` 个样本作为初始训练集

2. **训练阶段** (每个周期)
   - 将训练数据分词为模型可处理的序列
   - 训练 decoder-only Transformer 模型 `max_steps` 步
   - 模型学习预测给定前序 token 的下一个 token
   - 捕获高质量样本中存在的模式

3. **采样阶段** (每个周期)
   - 从训练好的模型采样 `num_samples_from_model` 个新序列
   - 使用温度控制采样保证多样性
   - 将序列解码回问题特定的对象

4. **局部搜索阶段** (每个周期)
   - 应用局部搜索修复采样对象的约束违反
   - 可选地用额外贪心步骤改进有效样本
   - 对所有处理后的样本评分

5. **选择阶段** (每个周期)
   - 合并新样本与现有训练数据
   - 若 `keep_only_unique=True` 则去重
   - 按分数选择前 `pop_size` 个样本
   - 成为下一周期的训练数据

6. **温度调整**
   - 若生成太多重复样本，按 `inc_temp` 增加温度
   - 在模型收敛时鼓励更多探索

---

## 💻 安装指南

### 系统要求

- Python 3.8+
- CUDA 兼容 GPU (推荐) 或 CPU
- 至少 8GB RAM (推荐 16GB+)
- 至少 10GB 可用磁盘空间

### 详细安装步骤

#### 方式 1: 使用 Micromamba (推荐)

```bash
# 安装 micromamba (如果尚未安装)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba

# 创建并激活环境
micromamba env create -f environment.yml
conda activate env_axplorer
```

#### 方式 2: 使用 Conda

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate env_axplorer
```

#### 验证安装

```bash
# 测试 PyTorch 是否可用
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 测试 CUDA (如果有 GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎓 训练模型

### 基础训练

从零开始训练模型（以 Turan 问题为例）：

```bash
python train.py \
    --env_name square \
    --exp_name my_first_exp \
    --N 30 \
    --encoding_tokens single_integer \
    --max_len 100 \
    --temperature '0.6'
```

### 恢复中断的训练

如果训练中断，可以从中断点继续：

```bash
python train.py \
    --env_name square \
    --exp_name my_first_exp \
    --exp_id 2026_01_22_18_51_53 \  # 指定之前的实验 ID
    --N 30 \
    --encoding_tokens single_integer \
    --max_len 100 \
    --temperature '0.6'
```

**原理**: 模型和优化器检查点存储在实验文件夹中，可以轻松恢复。

### 使用高质量数据

分两步走：先生成数据，再训练模型。适用于需要大量数据或数据生成成本高的场景。

**步骤 1: 生成数据**

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --gensize 10000000 \
    --pop_size 10000000 \
    --data_generation_only true
```

这将生成并保存 1000 万个样本。

**步骤 2: 只保留最优样本**

```bash
python train.py \
    --env_name square \
    --exp_name square_exp \
    --N 30 \
    --gensize 10000000 \
    --pop_size 100000 \
    --data_generation_only true
```

这将生成 1000 万但只保留最好的 10 万个。

⚠️ **重要提示**: 生成数据后，请复制 train/test 数据到新实验文件夹。如果不复制，训练任务会**覆盖**数据！

---

## 🖥️ Web UI 仪表盘

Axplorer 包含交互式 Web 仪表盘，用于监控训练进度和管理实验。

### 快速启动

```bash
# 激活环境
conda activate env_axplorer

# 启动 Web UI
python ui_dashboard.py

# 浏览器打开 http://localhost:7860
```

### 核心功能

#### 📈 训练监控
- **实时指标**: 查看实时训练指标（loss、最高分、测试分）
- **交互式图表**: 可视化平均分数、中位分数、最高分、Top 1% 分数的训练曲线
- **资源监控**: 跟踪训练时的 CPU 和内存使用率

#### 📁 实验管理
- **浏览所有实验**: 自动扫描并显示 `checkpoint/` 目录下的所有实验
- **实验详情**: 查看最终分数、状态（完成/运行中）、创建时间
- **日志查看器**: 直接从 UI 检查训练日志
- **可视化**: 一键生成训练图表

#### 🔧 配置管理
- **参数调优**: 调整温度、批次大小等超参数
- **恢复训练**: 轻松重启中断的实验
- **数据生成**: 配置数据生成参数

### 使用示例

#### 1. 查看最新训练

1. 切换到 "📈 训练监控" 标签页
2. 点击 "📊 生成训练图表"
3. 自动显示最新实验的训练曲线

#### 2. 管理实验

1. 切换到 "📁 实验管理" 标签页
2. 点击 "🔄 刷新实验列表"
3. 从表格中选择任意实验
4. 点击 "👁️ 查看日志" 或 "📊 生成图表"

#### 3. 比较结果

使用实验表格比较不同运行的最终分数  
一眼识别最佳配置

### 故障排除

如果 UI 没有显示你的实验：
- 确保实验以 `checkpoint/<exp_name>/<exp_id>/` 格式存储
- 检查实验目录中是否存在 `train.log` 或 `metrics.txt` 文件
- 点击 "🔄 刷新" 重新扫描 checkpoint 目录

详细文档见 [WEB_UI_FIXED.md](WEB_UI_FIXED.md) 和 [UI_USAGE_GUIDE.md](UI_USAGE_GUIDE.md)。

---

## 🔬 可用环境

| 环境名称 | 描述 | `env_name` | 数学问题 |
|----------|------|------------|----------|
| **无平方图** | 最大化无 4-环图的边数 | `square` | Turán 问题 |
| **无等腰三角形点集** | 最大化网格 [N]^2 中无等腰三角形的点数 | `isosceles` | 组合几何 |
| **无 5 点共球点集** | 最大化网格 [N]^3 中无 5 点共球的点数 | `sphere` | 离散几何 |
| **超立方体直径** | 寻找 d 维超立方体的最小边生成子图且直径为 d | `hypercube` | 图论优化 |

### 环境详细说明

#### 1. Square-free Graphs (`square`)

**问题**: 在 n 个顶点的图中，最多能有多少条边而不包含 4-环（C4）？

**数学背景**: 这是经典的 Turán 型问题，由 Erdős、Kővári、Sós 等人研究。

**编码方式**: 
- 每条边编码为单个整数：`N*i + j`
- 词汇表大小：C(N,2) = N*(N-1)/2

**示例命令**:
```bash
python train.py \
    --env_name square \
    --N 30 \
    --exp_name square_free_exp
```

#### 2. Isosceles-free Point Sets (`isosceles`)

**问题**: 在 N×N 网格中，最多能放置多少个点，使得任意三点不构成等腰三角形？

**数学背景**: 这是一个困难的组合几何问题，与 Erdős 不同的距离问题相关。

**编码方式**:
- 每个点编码为：`N*i + j`
- 序列长度：最多 N² 个点

#### 3. Sphere Point Sets (`sphere`)

**问题**: 在 N×N×N 网格中，最多能放置多少个点，使得任意 5 个点不在同一个球面上？

**数学背景**: 三维组合几何问题，涉及球面几何和离散数学。

**编码方式**:
- 每个点编码为：`N²*i + N*j + k`
- 坐标范围：[0, N-1]³

#### 4. Hypercube Diameter (`hypercube`)

**问题**: 找到 d 维超立方体的生成子图，保持直径为 d 的同时最小化边数。

**数学背景**: Graham-Harary 猜想已解决的图论优化问题。

**编码方式**:
- 每条边是顶点对 (i,j)，其中顶点编号 0 到 2^d-1
- 使用邻接矩阵表示

**示例命令**:
```bash
python train.py \
    --env_name hypercube \
    --N 5 \
    --exp_name hypercube_diameter_exp
```

---

## 🛠️ 自定义问题

要在自己的数学问题上使用 Axplorer，需要创建对应的环境。参考 `new_envs.ipynb` 中的分步指南和示例。

### 环境组件

1. **DataPoint 类**: 定义问题特定的数据结构
2. **Environment 类**: 实现评分函数、局部搜索、数据生成
3. **Tokenizer**: 将对象编码为 token 序列
4. **注册**: 在 `src/envs/__init__.py` 中注册新环境

### 快速模板

```python
# src/envs/my_problem.py
from src.envs.environment import DataPoint, BaseEnvironment

class MyDataPoint(DataPoint):
    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = ...  # 你的数据结构
        
        if init:
            self._initialize()
    
    def calc_score(self):
        """计算样本分数"""
        # 实现你的优化目标
        self.score = ...
    
    def calc_features(self):
        """生成用于去重的特征字符串"""
        self.features = ...

class MyEnvironment(BaseEnvironment):
    data_class = MyDataPoint
    k = 2  # 问题特定维度
    are_coordinates_symmetric = True  # 是否对称
```

详细教程请参考 `new_envs.ipynb`。

---

## ⚙️ 参数详解

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gensize` | 100000 | 初始数据生成数量 |
| `max_epochs` | 100 | 最大训练周期数 |
| `max_steps` | 50000 | 每个周期的训练步数 |
| `num_samples_from_model` | 300000 | 每个周期从模型采样的数量 |
| `pop_size` | 150000 | 每个周期后保留的样本数 |
| `ntest` | 1000 | 测试集大小 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_layer` | 8 | Transformer 层数 |
| `n_embd` | 512 | 嵌入向量维度 |
| `n_head` | 16 | 注意力头数 |
| `max_len` | 500 | 模型支持的最大序列长度 |
| `no_positional` | False | 是否禁用位置编码（置换不变问题时设为 True） |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 训练批次大小 |
| `learning_rate` | 5e-4 | AdamW 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `accumulation_steps` | 1 | 梯度累积步数（模拟大批次） |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.6 | 初始采样温度 |
| `temp_span` | 0 | 采样温度跨度 |
| `inc_temp` | 0.1 | 生成太多重复时的温度增量 |
| `top_k` | 50 | Top-k 采样（-1 表示禁用） |
| `keep_only_unique` | True | 是否只保留唯一样本 |

### 局部搜索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `always_search` | False | 总是应用局部搜索改进样本 |
| `redeem_only` | False | 只修复无效样本 |
| `process_pool` | False | 是否使用进程池加速评分 |
| `num_workers` | 0 | 工作进程数量 |

### 环境特定参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `k` | 问题特定维度 | Turan 问题 k=2（边对） |
| `are_coordinates_symmetric` | 坐标是否可置换 | 无向图 k=2 时为 True |
| `encoding_tokens` | 编码方式 | `single_integer`, `sequence_k_tokens`, `adjacency` |
| `make_object_canonical` | 是否规范化对象 | 去重时需要 |
| `augment_data_representation` | 是否增强数据表示 | 提升模型鲁棒性 |

---

## ❓ 常见问题

### Q1: 训练需要多长时间？

取决于问题规模和配置。典型配置下（N=30, 8 层 Transformer）：
- **初始数据生成**: 1-3 小时
- **每个周期**: 30 分钟 -1 小时
- **完整训练** (100 周期): 约 1-3 天

使用 GPU 可显著加速（见 [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md)）。

### Q2: 如何选择合适的 N？

- **小规模** (N<20): 快速原型验证，几分钟出结果
- **中规模** (N=20-40): 平衡质量和速度，推荐入门
- **大规模** (N>40): 需要更多资源和时间

建议从 N=30 开始，熟悉后再调整。

### Q3: 为什么采样显示 0 个样本生成？

可能原因：
1. **词汇表大小不匹配**: Tokenizer 基于错误的 N 构建
2. **解码参数错误**: 检查 `encoding_tokens` 配置
3. **所有样本都无效**: 评分函数过于严格

解决方法：查看日志中的 "Invalid examples" 统计。

### Q4: 如何加速训练？

1. **使用 GPU**: 参见 [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md)
2. **减少周期数**: 设置 `--max_epochs 10` 快速测试
3. **减小模型**: 降低 `n_layer`, `n_embd`
4. **禁用多进程**: 对于复杂评分任务，单进程更稳定

### Q5: 训练 Loss 为 0 怎么办？

这通常意味着**过拟合**或**数据问题**：
- **只有 1 个训练样本**: 检查数据生成 pipeline
- **所有样本相同**: 去重逻辑有问题
- **模型记住数据**: 增加数据多样性

详细诊断见 [HYPERCUBE_LOSS_ZERO_FINAL_FIX.md](HYPERCUBE_LOSS_ZERO_FINAL_FIX.md)。

### Q6: 如何可视化结果？

使用 Web UI 是最简单的方式：
```bash
python ui_dashboard.py
```

或者手动分析日志：
```bash
grep "Score distribution" checkpoint/<exp_name>/<exp_id>/train.log
```

---

## 📚 进阶主题

### 性能优化

- [GPU 加速完全指南](GPU_ACCELERATION_GUIDE.md) - 最大化 GPU 利用率
- [超立方体性能优化](HYPERCUBE_PERFORMANCE_OPTIMIZATION_COMPLETE.md) - 离散优化任务的陷阱规避

### 问题特定指南

- [超立方体环境详解](HYPERCUBE_ENV_GUIDE.md) - 从环境构建到训练
- [PatternBoost 突破解析](PATTERNBOOST_BREAKTHROUGH_EXPLAINED.md) - 核心算法洞察

### 开发文档

- [代码库总结](AXPLORER_CODE_SUMMARY.md) - 整体架构概览
- [数据生成问答](DATA_GENERATION_QA.md) - 深入理解数据流

---

## 🔗 相关资源

### 论文引用

如果你在工作中使用 Axplorer，请引用：

```bibtex
@article{charton2024transformers,
  title={Transformers learn optimal constructions of combinatorial objects through patternboost},
  author={Charton, Fran{\c{c}}ois and Ellenberg, Jordan S and Wagner, Adam Zsolt and Williamson, Geordie},
  journal={arXiv preprint arXiv:2403.08318},
  year={2024}
}
```

### 原始实现

PatternBoost 的原始代码由 François Charton, Jordan S. Ellenberg, Adam Zsolt Wagner, 和 Geordie Williamson 编写，可在 [这里](https://github.com/zawagner22/transformers_math_experiments) 找到。

### 社区与支持

- **GitHub Issues**: 报告 bug 或请求功能
- **讨论区**: 分享你的发现和技巧
- **邮件列表**: 获取最新更新

---

## 📄 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

感谢所有为 Axplorer 做出贡献的研究人员和开发者。这个项目建立在 PatternBoost 的基础上，旨在使 AI 辅助数学发现更加普及和易用。

---

<div align="center">

**开始你的数学发现之旅！** 🚀

[开始使用](#快速开始) · [查看示例](#可用环境) · [阅读文档](#进阶主题)

</div>

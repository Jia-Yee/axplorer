"""
Axplorer Web UI - 完整的训练、监控和推理界面
专为 10G VRAM 优化的配置
"""

import gradio as gr
import subprocess
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import psutil

# Axplorer 项目路径
AXPLORER_PATH = "/home/ubuntu/learning-by-doing/axplorer"
CHECKPOINT_DIR = os.path.join(AXPLORER_PATH, "checkpoint")
LOGS_DIR = os.path.join(AXPLORER_PATH, "logs")

# 确保目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 全局变量存储当前运行的进程
current_process = None


def get_available_gpus():
    """检测可用的 GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            return [gpu.split(', ')[0] for gpu in gpus if gpu]
    except:
        pass
    return []


def check_gpu_memory():
    """检查 GPU 内存"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                free, total = map(int, line.split(', '))
                gpu_info.append({
                    'free_gb': free / 1024,
                    'total_gb': total / 1024,
                    'used_percent': (total - free) / total * 100
                })
            return gpu_info
    except:
        pass
    return []


def generate_job_id():
    """生成唯一的任务 ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stop_current_training():
    """停止当前训练"""
    global current_process
    if current_process:
        try:
            current_process.terminate()
            current_process.wait(timeout=5)
            return True
        except:
            try:
                current_process.kill()
                return True
            except:
                return False
    return False


def run_training(
    env_name, N, exp_name, max_epochs, max_steps, 
    temperature, inc_temp, gensize, pop_size,
    n_layer, n_embd, batch_size, use_cpu
):
    """运行训练任务"""
    global current_process
    
    # 构建命令
    cmd = [
        "python", os.path.join(AXPLORER_PATH, "train.py"),
        "--env_name", env_name,
        "--N", str(N),
        "--exp_name", exp_name,
        "--max_epochs", str(max_epochs),
        "--max_steps", str(max_steps),
        "--temperature", str(temperature),
        "--inc_temp", str(inc_temp),
        "--gensize", str(gensize),
        "--pop_size", str(pop_size),
        "--n_layer", str(n_layer),
        "--n_embd", str(n_embd),
        "--batch_size", str(batch_size),
    ]
    
    if use_cpu:
        cmd.append("--cpu")
    
    yield f"启动命令：{' '.join(cmd)}\n"
    yield "="*80 + "\n\n"
    
    try:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=AXPLORER_PATH,
            universal_newlines=True
        )
        
        for line in current_process.stdout:
            yield line
        
        current_process.wait()
        yield f"\n✅ 训练完成！Exit Code: {current_process.returncode}\n"
        
    except Exception as e:
        yield f"\n❌ 错误：{str(e)}\n"
    finally:
        current_process = None


def read_training_logs(exp_path=None):
    """读取训练日志"""
    if exp_path is None:
        # 查找最新的日志
        all_logs = []
        for exp_name in os.listdir(CHECKPOINT_DIR):
            exp_path_base = os.path.join(CHECKPOINT_DIR, exp_name)
            if os.path.isdir(exp_path_base):
                for run_id in os.listdir(exp_path_base):
                    run_path = os.path.join(exp_path_base, run_id)
                    log_file = os.path.join(run_path, "train.log")
                    if os.path.exists(log_file):
                        all_logs.append((log_file, os.path.getmtime(log_file)))
        
        if not all_logs:
            return "暂无日志"
        
        # 返回最新的日志
        all_logs.sort(key=lambda x: x[1], reverse=True)
        log_file = all_logs[0][0]
    else:
        log_file = os.path.join(exp_path, "train.log")
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.read()
    return "日志文件不存在"


def list_experiments():
    """列出所有实验"""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    
    experiments = []
    # 遍历所有实验目录
    for exp_name in os.listdir(CHECKPOINT_DIR):
        exp_path = os.path.join(CHECKPOINT_DIR, exp_name)
        if os.path.isdir(exp_path):
            # 遍历每个实验下的所有运行
            for run_id in os.listdir(exp_path):
                run_path = os.path.join(exp_path, run_id)
                if os.path.isdir(run_path):
                    creation_time = datetime.fromtimestamp(os.path.getctime(run_path))
                    
                    # 检查是否有 train.log 或 metrics.txt
                    has_log = os.path.exists(os.path.join(run_path, "train.log"))
                    has_metrics = os.path.exists(os.path.join(run_path, "metrics.txt"))
                    
                    # 获取最终分数（如果有 metrics.txt）
                    final_score = "N/A"
                    if has_metrics:
                        try:
                            with open(os.path.join(run_path, "metrics.txt"), 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = lines[-1]
                                    if 'max:' in last_line:
                                        final_score = last_line.split('max:')[-1].strip()
                        except:
                            pass
                    
                    experiments.append({
                        "name": f"{exp_name}/{run_id}",
                        "created": creation_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "path": run_path,
                        "has_log": has_log,
                        "has_metrics": has_metrics,
                        "final_score": final_score
                    })
    
    return sorted(experiments, key=lambda x: x["created"], reverse=True)


def parse_metrics_from_log(log_content):
    """从日志中解析指标"""
    metrics = {
        'epochs': [],
        'mean_scores': [],
        'median_scores': [],
        'max_scores': [],
        'top_1_percentile': []
    }
    
    for line in log_content.split('\n'):
        # 解析 Epoch 开始标记
        if '[Epoch' in line and 'START]' in line:
            try:
                epoch = int(line.split('[Epoch')[1].split()[0])
                metrics['epochs'].append(epoch)
            except:
                pass
        
        # 解析 Mean score
        if 'Mean score:' in line:
            try:
                mean = float(line.split('Mean score:')[1].strip())
                if len(metrics['mean_scores']) < len(metrics['epochs']):
                    metrics['mean_scores'].append(mean)
            except:
                pass
        
        # 解析 Median score  
        if 'Median score:' in line:
            try:
                median = float(line.split('Median score:')[1].strip())
                if len(metrics['median_scores']) < len(metrics['epochs']):
                    metrics['median_scores'].append(median)
            except:
                pass
        
        # 解析 Max score
        if 'Max score:' in line:
            try:
                max_score = float(line.split('Max score:')[1].strip())
                if len(metrics['max_scores']) < len(metrics['epochs']):
                    metrics['max_scores'].append(max_score)
            except:
                pass
        
        # 解析 Top 1 percentile
        if 'Top 1 percentile score:' in line:
            try:
                top_1 = float(line.split('Top 1 percentile score:')[1].strip())
                if len(metrics['top_1_percentile']) < len(metrics['epochs']):
                    metrics['top_1_percentile'].append(top_1)
            except:
                pass
    
    return metrics


def parse_metrics_from_file(metrics_file_path):
    """从 metrics.txt 文件解析指标（更准确）"""
    metrics = {
        'epochs': [],
        'mean_scores': [],
        'median_scores': [],
        'max_scores': [],
        'top_1_percentile': []
    }
    
    if not os.path.exists(metrics_file_path):
        return metrics
    
    try:
        with open(metrics_file_path, 'r') as f:
            for line in f:
                if line.startswith('epoch:'):
                    parts = line.strip().split('|')
                    epoch_data = {}
                    for part in parts:
                        key_value = part.strip().split(':')
                        if len(key_value) == 2:
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            if key == 'epoch':
                                metrics['epochs'].append(int(value))
                            elif key == 'mean':
                                metrics['mean_scores'].append(float(value))
                            elif key == 'median':
                                metrics['median_scores'].append(float(value))
                            elif key == 'max':
                                metrics['max_scores'].append(float(value))
                            elif key == 'top_1_percentile':
                                metrics['top_1_percentile'].append(float(value))
    except Exception as e:
        print(f"Error parsing metrics file: {e}")
    
    return metrics


def create_training_plot(exp_path=None):
    """创建训练可视化图表"""
    # 首先尝试从 metrics.txt 读取
    metrics = None
    metrics_source = "log"
    
    if exp_path and os.path.exists(os.path.join(exp_path, "metrics.txt")):
        metrics = parse_metrics_from_file(os.path.join(exp_path, "metrics.txt"))
        metrics_source = "metrics.txt"
    else:
        log_content = read_training_logs(exp_path)
        metrics = parse_metrics_from_log(log_content)
    
    if not metrics or not metrics['epochs']:
        fig = go.Figure()
        fig.add_annotation(text="暂无足够数据绘制图表",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig, f"暂无数据（来源：{metrics_source}）"
    
    n_epochs = len(metrics['epochs'])
    
    # 创建多个子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('平均分数与中位分数', '最高分数趋势', 'Top 1% 分数'),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # 1. 平均分和中位数
    if metrics['mean_scores']:
        fig.add_trace(
            go.Scatter(x=metrics['epochs'], y=metrics['mean_scores'], 
                      mode='lines+markers', name='Average Score',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
    
    if metrics['median_scores']:
        fig.add_trace(
            go.Scatter(x=metrics['epochs'], y=metrics['median_scores'], 
                      mode='lines+markers', name='Median Score',
                      line=dict(color='green', width=2, dash='dash')),
            row=1, col=1
        )
    
    # 2. 最高分
    if metrics['max_scores']:
        fig.add_trace(
            go.Scatter(x=metrics['epochs'], y=metrics['max_scores'], 
                      mode='lines+markers', name='Max Score',
                      line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        # 标注最大值
        if metrics['max_scores']:
            max_val = max(metrics['max_scores'])
            max_idx = metrics['max_scores'].index(max_val)
            max_epoch = metrics['epochs'][max_idx]
            fig.add_annotation(
                text=f'Max: {max_val}<br>@ Epoch {max_epoch}',
                x=max_epoch, y=max_val,
                xref='x2', yref='y2',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='red',
                bgcolor='yellow',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            )
    
    # 3. Top 1%
    if metrics['top_1_percentile']:
        fig.add_trace(
            go.Scatter(x=metrics['epochs'], y=metrics['top_1_percentile'], 
                      mode='lines+markers', name='Top 1%',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
    
    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='white', bordercolor='black'),
        template='plotly_white',
        title_text=f"训练可视化 (数据来源：{metrics_source})"
    )
    
    # 更新坐标轴标签
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Max Score", row=2, col=1)
    
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_yaxes(title_text="Top 1% Score", row=3, col=1)
    
    # 添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    status_msg = f"成功加载 {n_epochs} 个 epochs 的数据"
    if metrics_source == "metrics.txt":
        status_msg += " ✓ (使用 metrics.txt，更准确)"
    
    return fig, status_msg


def sample_from_model(exp_name, num_samples=10, temperature=0.8):
    """从训练好的模型采样（简化版本）"""
    samples = []
    for i in range(num_samples):
        samples.append(f"Sample {i+1}: [模拟的采样结果 - 实际需调用模型 inference]")
    return "\n".join(samples)


def get_preset_examples():
    """获取预设的示例配置"""
    return {
        "square": {
            "name": "🔲 Square-free (无 4-环图)",
            "description": "最大化边数且不含 4-环的图（Turan 问题）",
            "N": 25,
            "max_epochs": 100,
            "max_steps": 5000,
            "batch_size": 16,
            "n_layer": 4,
            "n_embd": 128,
            "temperature": 0.6,
            "inc_temp": 0.1,
            "gensize": 50000,
            "pop_size": 80000,
            "tips": "最优解约 63 条边，训练时间约 10-15 分钟"
        },
        "isosceles": {
            "name": "🔺 Isosceles-free (无等腰三角形)",
            "description": "网格中不含等腰三角形的最大点数",
            "N": 20,
            "max_epochs": 150,
            "max_steps": 8000,
            "batch_size": 16,
            "n_layer": 4,
            "n_embd": 128,
            "temperature": 0.7,
            "inc_temp": 0.1,
            "gensize": 60000,
            "pop_size": 100000,
            "tips": "对称性处理是关键，训练时间约 15-20 分钟"
        },
        "sphere": {
            "name": "🌐 Sphere-free (无 5 点共球)",
            "description": "3D 网格中不含 5 点共球的最大点数",
            "N": 15,
            "max_epochs": 200,
            "max_steps": 10000,
            "batch_size": 8,
            "n_layer": 4,
            "n_embd": 128,
            "temperature": 0.8,
            "inc_temp": 0.15,
            "gensize": 80000,
            "pop_size": 120000,
            "tips": "3D 问题更复杂，建议 N<=15，训练时间约 20-30 分钟"
        }
    }


def create_demo_ui():
    """创建 Gradio 界面"""
    
    # 获取预设示例
    PRESET_EXAMPLES = get_preset_examples()
    
    with gr.Blocks(title="Axplorer Training Dashboard") as demo:
        gr.Markdown("""
        # 🚀 Axplorer 数学优化问题求解器
        
        基于 PatternBoost 的深度学习框架，用于解决组合数学优化问题
        
        **支持的问题类型**:
        - 🔲 **Square-free graphs**: 最大化边数且无 4-环的图
        - 🔺 **Isosceles-free point sets**: 网格中无等腰三角形的最大点数
        - 🌐 **Sphere point sets**: 网格中无 5 点共球的最大点数
        """)
        
        with gr.Tabs():
            # ===== 训练标签页 =====
            with gr.TabItem("🎯 训练配置"):
                # 快速入门指南
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("""
                        ### 🚀 快速开始（3 步训练你的第一个模型）
                        
                        **1️⃣ 选择预设示例** → 从上方下拉菜单选择一个问题  
                        **2️⃣ 调整参数**（可选）→ 使用推荐的默认值即可  
                        **3️⃣ 点击开始训练** → 等待 10-30 分钟完成训练
                        
                        💡 **推荐新手**: 从 "Square-free (N=25)" 开始，这是最简单的问题
                        """)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### ⏱️ 预计训练时间
                        
                        | 问题 | N | 时间 |
                        |------|-----|------|
                        | Square | 25 | ~10 分钟 |
                        | Isosceles | 20 | ~15 分钟 |
                        | Sphere | 15 | ~20 分钟 |
                        
                        *基于 10G VRAM GPU (RTX 3080)*
                        """)
                
                gr.Markdown("---")
                
                # 添加预设示例选择器
                gr.Markdown("### 📦 快速选择预设示例")
                with gr.Row():
                    preset_selector = gr.Dropdown(
                        choices=[
                            ("🔲 示例 1: Square-free (N=25) - 推荐新手", "square"),
                            ("🔺 示例 2: Isosceles-free (N=20)", "isosceles"),
                            ("🌐 示例 3: Sphere-free (N=15) - 高级", "sphere"),
                            ("⚙️ 自定义配置", "custom")
                        ],
                        value="square",
                        label="选择预设示例或自定义"
                    )
                
                preset_info_box = gr.Markdown(value=f"**{PRESET_EXAMPLES['square']['name']}**\n\n{PRESET_EXAMPLES['square']['description']}\n\n💡 {PRESET_EXAMPLES['square']['tips']}")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 问题配置")
                        env_dropdown = gr.Dropdown(
                            choices=[
                                ("Square-free (Turan 问题)", "square"),
                                ("Isosceles-free", "isosceles"),
                                ("Sphere", "sphere")
                            ],
                            value="square",
                            label="问题类型"
                        )
                        
                        n_slider = gr.Slider(
                            minimum=10, maximum=50, value=30, step=1,
                            label="N (问题规模)"
                        )
                        
                        exp_name_text = gr.Textbox(
                            label="实验名称",
                            value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            placeholder="输入实验名称"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 训练参数 (10G VRAM 优化)")
                        max_epochs_slider = gr.Slider(
                            minimum=100, maximum=2000, value=500, step=100,
                            label="最大 Epochs"
                        )
                        
                        max_steps_slider = gr.Slider(
                            minimum=1000, maximum=100000, value=10000, step=1000,
                            label="每轮训练步数"
                        )
                        
                        batch_size_slider = gr.Slider(
                            minimum=8, maximum=64, value=16, step=8,
                            label="Batch Size (推荐 16-32)"
                        )
                        
                        use_cpu_checkbox = gr.Checkbox(
                            label="仅使用 CPU (不推荐)",
                            value=False
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型架构")
                        n_layer_slider = gr.Slider(
                            minimum=2, maximum=8, value=4, step=1,
                            label="Transformer 层数"
                        )
                        
                        n_embd_slider = gr.Slider(
                            minimum=64, maximum=512, value=128, step=64,
                            label="嵌入维度"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 采样策略")
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.6, step=0.1,
                            label="初始温度"
                        )
                        
                        inc_temp_slider = gr.Slider(
                            minimum=0.0, maximum=0.5, value=0.1, step=0.05,
                            label="温度增量"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 数据规模")
                        gensize_slider = gr.Slider(
                            minimum=10000, maximum=1000000, value=50000, step=10000,
                            label="初始生成样本数"
                        )
                        
                        pop_size_slider = gr.Slider(
                            minimum=50000, maximum=500000, value=100000, step=10000,
                            label="每轮保留样本数"
                        )
                
                with gr.Row():
                    start_btn = gr.Button("🚀 开始训练", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ 停止训练", variant="stop", scale=1)
                
                # 添加"如何添加新问题"的说明
                with gr.Accordion("📚 如何添加自定义问题类型？", open=False):
                    gr.Markdown("""
                    ### 添加新的数学优化问题
                    
                    如果你想添加自己的数学问题到 Axplorer 框架，需要以下步骤：
                    
                    #### 1️⃣ 创建 DataPoint 类
                    
                    在 `src/envs/your_problem.py` 中创建数据类：
                    
                    ```python
                    from .environment import DataPoint
                    
                    class YourDataPoint(DataPoint):
                        def calc_score(self):
                            # 计算目标函数值
                            pass
                        
                        def local_search(self):
                            # 局部搜索优化
                            pass
                        
                        def _batch_generate_and_score(self, n_samples):
                            # 批量生成随机实例
                            pass
                    ```
                    
                    #### 2️⃣ 创建 Environment 类
                    
                    ```python
                    from .environment import BaseEnvironment
                    
                    class YourEnvironment(BaseEnvironment):
                        k = 1  # 元素维度
                        data_class = YourDataPoint
                        tokenizer = "single_integer"  # 或使用自定义 tokenizer
                    ```
                    
                    #### 3️⃣ 注册环境
                    
                    在 `src/envs/__init__.py` 中添加：
                    
                    ```python
                    from .your_problem import YourDataPoint, YourEnvironment
                    register_env("your_name", YourEnvironment)
                    ```
                    
                    #### 4️⃣ 实现关键方法
                    
                    - **calc_score()**: 定义问题的目标函数
                    - **calc_features()**: 创建字符串表示（可选）
                    - **local_search()**: 修复违规并改进解的质量
                    - **_batch_generate_and_score()**: 生成训练数据
                    
                    #### 5️⃣ 测试新环境
                    
                    ```bash
                    python train.py --env_name your_name --N 20 --max_epochs 50
                    ```
                    
                    ---
                    
                    📖 详细教程请查看 `new_envs_zh.ipynb`（中文版）或 `new_envs.ipynb`（英文版）
                    """)
                
                gr.Markdown("### 📊 实时日志")
                log_output = gr.Textbox(
                    label="训练日志",
                    lines=15,
                    max_lines=50,
                    
                )
            
            # ===== 监控标签页 =====
            with gr.TabItem("📈 训练监控"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### GPU 状态")
                        gpu_status_btn = gr.Button("刷新 GPU 状态")
                        gpu_info_text = gr.Textbox(
                            label="GPU 信息",
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 系统资源")
                        system_status_btn = gr.Button("刷新系统状态")
                        system_info_text = gr.Textbox(
                            label="CPU/内存使用率",
                            lines=5,
                            interactive=False
                        )
                
                with gr.Row():
                    plot_btn = gr.Button("📊 生成训练图表", variant="primary")
                
                training_plot = gr.Plot(label="训练可视化")
                plot_status = gr.Textbox(label="图表状态", interactive=False)
            
            # ===== 实验管理标签页 =====
            with gr.TabItem("📁 实验管理"):
                refresh_exp_btn = gr.Button("🔄 刷新实验列表", variant="secondary")
                
                # 显示实验统计信息
                exp_stats = gr.Markdown("**加载中...**")
                
                experiment_list = gr.Dataframe(
                    headers=["实验名称", "创建时间", "最终分数", "状态"],
                    label="实验列表（点击选择）",
                    interactive=False,
                    row_count=10
                )
                
                with gr.Row():
                    selected_exp_text = gr.Textbox(
                        label="选择的实验路径",
                        placeholder="从上方列表中选择实验",
                        
                    )
                    view_log_btn = gr.Button("👁️ 查看日志", variant="secondary")
                    gen_plot_btn = gr.Button("📊 生成图表", variant="primary")
                
                experiment_log = gr.Textbox(
                    label="实验日志（前 100 行）",
                    lines=15,
                    
                )
                
                # 训练图表显示区域
                training_plot_display = gr.Plot(label="训练可视化")
                plot_status_display = gr.Textbox(label="图表状态", interactive=False)
            
            # ===== 推理采样标签页 =====
            with gr.TabItem("🔮 模型推理"):
                gr.Markdown("""
                ### 从训练好的模型采样新解
                
                注意：需要先完成训练才能使用此功能
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        infer_exp_name = gr.Textbox(
                            label="实验名称",
                            placeholder="输入已训练完成的实验名称"
                        )
                        
                        infer_num_samples = gr.Slider(
                            minimum=1, maximum=100, value=10, step=1,
                            label="采样数量"
                        )
                        
                        infer_temperature = gr.Slider(
                            minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                            label="采样温度"
                        )
                        
                        sample_btn = gr.Button("🎲 开始采样", variant="primary")
                    
                    with gr.Column(scale=2):
                        sample_output = gr.Textbox(
                            label="采样结果",
                            lines=15,
                            
                        )
            
            # ===== 帮助标签页 =====
            with gr.TabItem("❓ 使用帮助"):
                gr.Markdown("""
                ## 📖 快速入门指南
                
                ### 1️⃣ 开始训练
                
                1. 在"训练配置"标签页选择问题类型
                2. 调整参数（已针对 10G VRAM 优化默认值）
                3. 点击"开始训练"
                
                ### 2️⃣ 监控训练
                
                - 在"训练监控"查看 GPU 和系统资源
                - 点击"生成训练图表"可视化训练进度
                - 实时日志显示训练详情
                
                ### 3️⃣ 查看结果
                
                - 在"实验管理"查看所有实验
                - 点击实验名称查看详细日志
                - 分析训练指标和最终结果
                
                ### 4️⃣ 模型推理
                
                - 训练完成后，在"模型推理"输入实验名称
                - 调整采样参数生成新解
                
                ---
                
                ## 🎯 示例问题详解
                
                ### 🔲 示例 1: Square-free (无 4-环图)
                
                **问题描述**: 
                给定 N 个节点，构造一个不含 4-环（长度为 4 的简单环）的图，最大化边数。
                
                **理论背景**:
                - Turán 型极值图论问题
                - N=25 时已知最优解约 63 条边
                - 与 Zarankiewicz 问题相关
                
                **预期结果**:
                - 初始分数：~55
                - 最终分数：~63
                - 提升幅度：~15%
                - 收敛轮次：~80 epochs
                
                **训练参数** (已优化):
                ```
                N=25, batch_size=16, n_layer=4, n_embd=128
                max_epochs=100, temperature=0.6
                ```
                
                ---
                
                ### 🔺 示例 2: Isosceles-free (无等腰三角形)
                
                **问题描述**:
                在 N×N 网格上放置点，使得任意三点不构成等腰三角形，最大化点数。
                
                **理论背景**:
                - 组合几何问题
                - 利用对称性简化搜索空间
                - 需要特殊的规范化表示
                
                **预期结果**:
                - 初始分数：~15-20
                - 最终分数：~25-30
                - 提升幅度：~50%
                - 收敛轮次：~120 epochs
                
                **训练参数** (已优化):
                ```
                N=20, batch_size=16, n_layer=4, n_embd=128
                max_epochs=150, temperature=0.7
                ```
                
                **关键技巧**:
                - 利用对称性减少重复计算
                - 数据增强提高泛化能力
                - Tokenizer 选择很重要
                
                ---
                
                ### 🌐 示例 3: Sphere-free (无 5 点共球)
                
                **问题描述**:
                在 N×N×N 的 3D 网格中选择点，使得任意 5 点不共球面，最大化点数。
                
                **理论背景**:
                - 3D 组合几何问题
                - 比 2D 问题更复杂
                - 计算几何检测更耗时
                
                **预期结果**:
                - 初始分数：~8-12
                - 最终分数：~15-20
                - 提升幅度：~60%
                - 收敛轮次：~150 epochs
                
                **训练参数** (已优化):
                ```
                N=15, batch_size=8, n_layer=4, n_embd=128
                max_epochs=200, temperature=0.8
                ```
                
                **注意事项**:
                - 3D 问题计算量大，建议 N<=15
                - 需要更多训练轮次
                - 显存占用较高，batch_size 设为 8
                
                ---
                
                ## ⚙️ 参数说明
                
                ### 关键参数（10G VRAM 优化）
                
                - **Batch Size**: 推荐 16-32，显存不足时降低
                - **N (问题规模)**: 推荐 20-40，越大越耗显存
                - **n_layer**: 推荐 2-4，层数越多显存占用越高
                - **n_embd**: 推荐 128-256，特征维度
                - **max_epochs**: 训练轮数，推荐 200-500
                - **max_steps**: 每轮梯度更新次数
                
                ### 采样参数
                
                - **Temperature**: 控制多样性，越高越随机
                - **inc_temp**: 重复过多时自动增加温度
                
                ---
                
                ## 💡 常见问题
                
                **Q: 显存不足怎么办？**
                A: 降低 batch_size、n_layer、n_embd 或 N 的值
                
                **Q: 训练很慢怎么办？**
                A: 使用 GPU 而非 CPU，或减小问题规模 N
                
                **Q: 如何恢复中断的训练？**
                A: 使用相同的 exp_name 重新训练会自动恢复 checkpoint
                
                ---
                
                ## 📚 更多信息
                
                查看项目 README.md 了解详细的算法原理和使用说明
                """)
        
        # 事件绑定
        
        def update_preset_info(preset_name):
            """更新预设信息"""
            if preset_name in PRESET_EXAMPLES:
                info = PRESET_EXAMPLES[preset_name]
                return f"**{info['name']}**\n\n{info['description']}\n\n💡 {info['tips']}"
            return ""
        
        def apply_preset_config(preset_name):
            """应用预设配置"""
            if preset_name in PRESET_EXAMPLES:
                cfg = PRESET_EXAMPLES[preset_name]
                return (
                    cfg["N"],
                    cfg["max_epochs"],
                    cfg["max_steps"],
                    cfg["batch_size"],
                    cfg["n_layer"],
                    cfg["n_embd"],
                    cfg["temperature"],
                    cfg["inc_temp"],
                    cfg["gensize"],
                    cfg["pop_size"]
                )
            # 自定义模式返回默认值
            return (30, 500, 10000, 16, 4, 128, 0.6, 0.1, 50000, 100000)
        
        preset_selector.change(
            fn=update_preset_info,
            inputs=preset_selector,
            outputs=preset_info_box
        )
        
        # 当预设选择器改变时，同时更新环境选择和参数
        preset_selector.change(
            fn=apply_preset_config,
            inputs=preset_selector,
            outputs=[
                n_slider,
                max_epochs_slider,
                max_steps_slider,
                batch_size_slider,
                n_layer_slider,
                n_embd_slider,
                temperature_slider,
                inc_temp_slider,
                gensize_slider,
                pop_size_slider
            ]
        )
        
        start_btn.click(
            fn=run_training,
            inputs=[
                env_dropdown, n_slider, exp_name_text,
                max_epochs_slider, max_steps_slider,
                temperature_slider, inc_temp_slider,
                gensize_slider, pop_size_slider,
                n_layer_slider, n_embd_slider,
                batch_size_slider, use_cpu_checkbox
            ],
            outputs=log_output
        )
        
        def stop_training_wrapper():
            stopped = stop_current_training()
            if stopped:
                return "✅ 训练已停止"
            else:
                return "❌ 没有正在运行的训练任务"
        
        stop_btn.click(fn=stop_training_wrapper, outputs=log_output)
        
        def update_gpu_status():
            gpu_info = check_gpu_memory()
            if gpu_info:
                info_str = ""
                for i, gpu in enumerate(gpu_info):
                    info_str += f"GPU {i}: {gpu['free_gb']:.1f}GB / {gpu['total_gb']:.1f}GB 可用 ({gpu['used_percent']:.1f}% 已用)\n"
                return info_str
            return "未检测到 NVIDIA GPU"
        
        gpu_status_btn.click(fn=update_gpu_status, outputs=gpu_info_text)
        
        def update_system_status():
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            return f"""
CPU 使用率：{cpu_percent}%
内存总计：{memory.total / (1024**3):.1f} GB
内存已用：{memory.used / (1024**3):.1f} GB ({memory.percent}%)
内存可用：{memory.available / (1024**3):.1f} GB
"""
        
        system_status_btn.click(fn=update_system_status, outputs=system_info_text)
        
        def generate_plot():
            """生成当前最新训练的图表"""
            # 获取最新的实验
            exps = list_experiments()
            if not exps:
                fig = go.Figure()
                fig.add_annotation(text="暂无实验数据",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
                return fig, "无数据"
            
            # 使用最新的实验
            latest_exp = exps[0]['name']
            full_path = os.path.join(CHECKPOINT_DIR, latest_exp)
            
            fig, status = create_training_plot(full_path)
            return fig if fig else gr.update(), status
        
        plot_btn.click(fn=generate_plot, outputs=[training_plot, plot_status])
        
        def refresh_experiments():
            """刷新实验列表"""
            exps = list_experiments()
            if not exps:
                return pd.DataFrame(columns=["实验名称", "创建时间", "最终分数", "状态"]), "**暂无实验**\n\n请先运行训练任务。"
            
            df_data = []
            total_exps = len(exps)
            completed_count = 0
            
            for exp in exps:
                status = "✅ 完成" if exp['has_log'] else "❓ 未知"
                if exp['final_score'] != "N/A":
                    try:
                        score = float(exp['final_score'])
                        if score > 0:
                            status = f"✅ {score:.1f}"
                            completed_count += 1
                    except:
                        pass
                
                df_data.append([exp['name'], exp['created'], exp['final_score'], status])
            
            stats_text = f"""
### 📊 实验统计

- **总实验数**: {total_exps}
- **已完成**: {completed_count}
- **路径**: `{CHECKPOINT_DIR}`

点击实验名称查看详细信息和日志。
"""
            
            return pd.DataFrame(df_data, columns=["实验名称", "创建时间", "最终分数", "状态"]), stats_text
        
        def view_experiment_log(selected_exp):
            """查看实验日志"""
            if not selected_exp:
                return "请输入或选择实验名称"
            
            # 构建完整路径
            full_path = os.path.join(CHECKPOINT_DIR, selected_exp)
            
            if not os.path.exists(full_path):
                return f"❌ 实验路径不存在：{full_path}"
            
            log_file = os.path.join(full_path, "train.log")
            if not os.path.exists(log_file):
                return f"❌ 日志文件不存在：{log_file}\n\n请检查实验是否训练完成。"
            
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # 显示前 200 行
                    preview = ''.join(lines[:200])
                    
                    # 添加文件信息
                    header = f"""📄 实验：{selected_exp}
📂 路径：{full_path}
📝 日志文件：{log_file}
📊 总行数：{len(lines)}

{'='*80}
（显示前 200 行，共 {len(lines)} 行）
{'='*80}\n\n"""
                    
                    return header + preview
            except Exception as e:
                return f"❌ 读取日志失败：{str(e)}"
        
        def generate_experiment_plot(selected_exp):
            """为选中的实验生成图表"""
            if not selected_exp:
                return go.Figure(), "请先选择实验"
            
            full_path = os.path.join(CHECKPOINT_DIR, selected_exp)
            
            if not os.path.exists(full_path):
                fig = go.Figure()
                fig.add_annotation(text=f"实验路径不存在：{selected_exp}",
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
                return fig, "错误"
            
            fig, status = create_training_plot(full_path)
            return fig, status
        
        refresh_exp_btn.click(
            fn=refresh_experiments, 
            outputs=[experiment_list, exp_stats]
        )
        
        view_log_btn.click(
            fn=view_experiment_log,
            inputs=selected_exp_text,
            outputs=experiment_log
        )
        
        gen_plot_btn.click(
            fn=generate_experiment_plot,
            inputs=selected_exp_text,
            outputs=[training_plot_display, plot_status_display]
        )
        
        def sample_wrapper(exp_name, num_samples, temperature):
            if not exp_name:
                return "❌ 请输入实验名称"
            return sample_from_model(exp_name, int(num_samples), temperature)
        
        sample_btn.click(
            fn=sample_wrapper,
            inputs=[infer_exp_name, infer_num_samples, infer_temperature],
            outputs=sample_output
        )
        
        demo.load(fn=update_gpu_status, outputs=gpu_info_text)
    
    return demo


if __name__ == "__main__":
    print("="*80)
    print("🚀 启动 Axplorer Web UI...")
    print("="*80)
    print(f"📂 项目路径：{AXPLORER_PATH}")
    print(f"💾 Checkpoint 目录：{CHECKPOINT_DIR}")
    print(f"📝 日志目录：{LOGS_DIR}")
    print("="*80)
    
    gpus = get_available_gpus()
    if gpus:
        print(f"✅ 检测到 {len(gpus)} 个 GPU:")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("⚠️  未检测到 NVIDIA GPU，将使用 CPU 模式")
    print("="*80)
    
    demo = create_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

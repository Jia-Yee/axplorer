"""
训练结果分析脚本 - 查看 demo_train.py 的运行结果
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_training_results(checkpoint_dir="checkpoint/demo_square_free"):
    """分析训练结果"""
    
    print("=" * 80)
    print("🎯 Axplorer 训练结果分析")
    print("=" * 80)
    print()
    
    # 找到最新的实验
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ 未找到检查点目录：{checkpoint_dir}")
        return
    
    # 获取所有实验文件夹
    exp_folders = sorted([f for f in checkpoint_path.iterdir() if f.is_dir()], 
                        key=lambda x: str(x), reverse=True)
    
    if not exp_folders:
        print("❌ 未找到任何实验文件夹")
        return
    
    latest_exp = exp_folders[0]
    print(f"📁 最新实验：{latest_exp.name}")
    print(f"📂 完整路径：{latest_exp}")
    print()
    
    # 读取 metrics.txt
    metrics_file = latest_exp / "metrics.txt"
    if not metrics_file.exists():
        print(f"❌ 未找到指标文件：{metrics_file}")
        return
    
    print("📊 加载训练指标...")
    epochs_data = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.startswith('epoch:'):
                # 解析行：epoch: 0 | mean: 54.65 | median: 55.0 | ...
                parts = line.strip().split('|')
                epoch_data = {}
                for part in parts:
                    key_value = part.strip().split(':')
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        if key == 'epoch':
                            epoch_data[key] = int(value)
                        else:
                            try:
                                epoch_data[key] = float(value)
                            except ValueError:
                                epoch_data[key] = value
                
                epochs_data.append(epoch_data)
    
    if not epochs_data:
        print("❌ 未能解析任何训练数据")
        return
    
    # 转换为 DataFrame
    df = pd.DataFrame(epochs_data)
    
    print(f"✅ 成功加载 {len(df)} 个 epoch 的数据")
    print()
    
    # 显示关键统计信息
    print("=" * 80)
    print("📈 训练统计摘要")
    print("=" * 80)
    print(f"总 Epoch 数：{len(df)}")
    print(f"初始分数 (Epoch 0):")
    print(f"  - 平均分：{df.iloc[0]['mean']:.2f}")
    print(f"  - 中位数：{df.iloc[0]['median']:.2f}")
    print(f"  - 最高分：{df.iloc[0]['max']}")
    print()
    print(f"最终分数 (Epoch {df.iloc[-1]['epoch']}):")
    print(f"  - 平均分：{df.iloc[-1]['mean']:.2f}")
    print(f"  - 中位数：{df.iloc[-1]['median']:.2f}")
    print(f"  - 最高分：{df.iloc[-1]['max']}")
    print()
    print(f"性能提升:")
    print(f"  - 平均分提升：{df.iloc[-1]['mean'] - df.iloc[0]['mean']:.2f} ({((df.iloc[-1]['mean']/df.iloc[0]['mean'])-1)*100:.1f}%)")
    print(f"  - 最高分提升：{df.iloc[-1]['max'] - df.iloc[0]['max']}")
    print()
    
    # 绘制训练曲线
    print("=" * 80)
    print("📊 生成可视化图表...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 平均分和中位数趋势
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['mean'], 'b-', label='Average Score', linewidth=2)
    ax1.plot(df['epoch'], df['median'], 'g--', label='Median Score', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Training Progress: Average & Median Scores', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 最高分趋势
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['max'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Best Score', fontsize=12)
    ax2.set_title('Best Score Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 添加标注
    max_score = df['max'].max()
    max_epoch = df.loc[df['max'].idxmax(), 'epoch']
    ax2.annotate(f'Max: {max_score}\n@ Epoch {max_epoch}',
                xy=(max_epoch, max_score),
                xytext=(max_epoch*0.7, max_score*0.95),
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 3. Top 1% 分数趋势
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['top_1_percentile'], 'm-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Top 1% Score', fontsize=12)
    ax3.set_title('Top 1% Percentile Score', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. 分数分布箱线图（每 10 个 epoch）
    ax4 = axes[1, 1]
    sample_epochs = df.iloc[::max(1, len(df)//10)].copy()
    
    # 计算误差范围，确保非负
    lower_error = sample_epochs['mean'] - sample_epochs['median']
    upper_error = sample_epochs['max'] - sample_epochs['mean']
    
    # 确保误差值为正
    lower_error = lower_error.abs()
    upper_error = upper_error.abs()
    
    ax4.bar(range(len(sample_epochs)), sample_epochs['mean'], 
           yerr=[lower_error, upper_error],
           capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax4.set_xlabel('Epoch Sample', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Score Distribution (Sampled)', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = latest_exp / "training_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存到：{plot_file}")
    print()
    
    # 显示其他输出文件
    print("=" * 80)
    print("📁 其他重要文件")
    print("=" * 80)
    
    files_to_check = [
        ("训练日志", "train.log"),
        ("模型权重", "model.pt"),
        ("优化器状态", "optimizer.pt"),
        ("Epoch 信息", "epoch.txt"),
        ("温度信息", "temperature.txt"),
    ]
    
    for desc, filename in files_to_check:
        filepath = latest_exp / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {desc}: {filename} ({size / 1024:.1f} KB)")
        else:
            print(f"✗ {desc}: {filename} (不存在)")
    
    print()
    print("=" * 80)
    print("💡 如何查看详细信息")
    print("=" * 80)
    print()
    print("1. 查看完整训练日志:")
    print(f"   tail -f {latest_exp}/train.log")
    print()
    print("2. 查看训练指标:")
    print(f"   cat {latest_exp}/metrics.txt")
    print()
    print("3. 打开可视化图表:")
    print(f"   eog {plot_file}  # Linux")
    print(f"   open {plot_file}  # macOS")
    print()
    print("4. 使用 Web UI 查看:")
    print("   访问 http://localhost:7860")
    print("   切换到'实验管理'标签页")
    print()
    
    # 返回分析结果
    return {
        'experiment': str(latest_exp),
        'epochs': len(df),
        'final_mean': df.iloc[-1]['mean'],
        'final_median': df.iloc[-1]['median'],
        'final_max': df.iloc[-1]['max'],
        'improvement': df.iloc[-1]['mean'] - df.iloc[0]['mean'],
        'dataframe': df
    }


if __name__ == "__main__":
    result = analyze_training_results()
    
    if result:
        print("\n" + "=" * 80)
        print("✨ 训练完成总结")
        print("=" * 80)
        print(f"实验名称：{result['experiment']}")
        print(f"训练轮数：{result['epochs']} epochs")
        print(f"最终平均分：{result['final_mean']:.2f}")
        print(f"最终最高分：{result['final_max']}")
        print(f"性能提升：+{result['improvement']:.2f}")
        print("=" * 80)

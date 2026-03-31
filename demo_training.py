"""
Axplorer 演示训练脚本 - 快速体验完整流程（10G VRAM 优化）
"""

import subprocess
import sys
from pathlib import Path

def run_demo():
    """运行一个快速的演示训练"""
    
    print("="*80)
    print("🎯 Axplorer 演示训练 - Square-free 问题")
    print("="*80)
    print()
    print("配置说明:")
    print("  • 问题类型：Square-free (无 4-环图)")
    print("  • 问题规模：N=25 (适中规模)")
    print("  • 训练轮数：100 epochs (快速演示)")
    print("  • Batch Size: 16 (10G VRAM 优化)")
    print("  • 模型大小：4 层，128 维嵌入")
    print()
    print("预计时间：~10-15 分钟")
    print("="*80)
    print()
    
    # 构建命令
    cmd = [
        sys.executable, "train.py",
        "--env_name", "square",
        "--exp_name", "demo_square_free",
        "--N", "25",
        "--max_epochs", "100",
        "--max_steps", "5000",
        "--batch_size", "16",
        "--n_layer", "4",
        "--n_embd", "128",
        "--gensize", "50000",
        "--pop_size", "80000",
        "--num_samples_from_model", "200000",
        "--temperature", "0.6",
        "--inc_temp", "0.1",
        "--keep_only_unique", "true"
    ]
    
    print(f"执行命令：{' '.join(cmd)}")
    print()
    print("-"*80)
    
    try:
        # 运行训练
        process = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            universal_newlines=True
        )
        
        print()
        print("-"*80)
        if process.returncode == 0:
            print("✅ 演示训练完成！")
            print()
            print("查看结果:")
            print(f"  📂 Checkpoint 目录：checkpoint/demo_square_free/")
            print(f"  📊 日志文件：logs/*.log")
            print()
            print("下一步:")
            print("  1. 在浏览器中访问 http://localhost:7860")
            print("  2. 在'实验管理'标签页查看训练结果")
            print("  3. 使用'模型推理'功能生成新解")
        else:
            print(f"❌ 训练失败，退出码：{process.returncode}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
    
    print("="*80)

if __name__ == "__main__":
    run_demo()

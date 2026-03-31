#!/bin/bash

echo "======================================"
echo "  Axplorer Web UI 启动脚本"
echo "======================================"

# 激活虚拟环境
source venv_axplorer/bin/activate

# 检查依赖
echo "检查依赖..."
python -c "import gradio; import plotly; import pandas; import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖，请运行：pip install gradio plotly pandas psutil matplotlib"
    exit 1
fi

echo "✅ 依赖检查通过"
echo ""
echo "启动 Web UI 服务器..."
echo "访问地址：http://localhost:7860"
echo "======================================"

# 启动 UI
python ui_dashboard.py


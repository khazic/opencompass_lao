#!/bin/bash

# 模型基本路径
MODEL_BASE="/xfr_ceph_sh/liuchonghan/paper_model"

DATASETS="mmlu_gen humaneval_gen cmmlu_gen bbh_gen hellaswag_gen winogrande_gen"
# 模型和对应的GPU
declare -A MODEL_GPU=(
    ["apertus-8b"]="1"
    ["aya-expanse-8b"]="2"
    ["emma-500"]="4"
    ["gamme3-27b"]="5"
)

# 创建输出目录
timestamp=$(date '+%Y%m%d_%H%M%S')
out_dir="outputs/benchmark_${timestamp}"
mkdir -p "logs" "$out_dir"

# 启动每个模型的评测
for model_name in "${!MODEL_GPU[@]}"; do
    gpu_id="${MODEL_GPU[$model_name]}"
    model_path="$MODEL_BASE/$model_name"
    
    echo "启动模型: $model_name 在 GPU $gpu_id"
    
    # 创建新的 tmux 会话
    tmux new-session -d -s "${model_name}" "
        # 设置环境变量
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export COMPASS_DATA_CACHE=/xfr_ceph_sh/liuchonghan/opencompass_lao
        export OPENCOMPASS_DATASETS_PATH=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export OPENCOMPASS_CACHE_DIR=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export COMPASS_ALLOW_DOWNLOAD=0
        
        # 运行评测
        python run.py \\
            --datasets $DATASETS \\
            --hf-path '$model_path' \\
            --hf-type chat \\
            --work-dir '$out_dir/${model_name}' \\
            > logs/${model_name}_${timestamp}.log 2>&1
            
        # 显示完成消息
        echo '$model_name 评测完成!' >> logs/completion_status.log
    "
    
    echo "$model_name 评测已在 tmux 会话中启动"
    sleep 2  # 短暂暂停，避免资源竞争
done

echo "所有模型评测已启动！"
echo "结果目录：$out_dir/"
echo "日志目录：logs/"
echo "使用 'tmux attach -t 模型名称' 可查看进度"
echo "使用 'cat logs/completion_status.log' 可查看完成状态"
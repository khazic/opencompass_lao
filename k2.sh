#!/bin/bash

MODEL_BASE="/xfr_ceph_sh/liuchonghan/paper_model"

# 使用 demo 中的数据集
DATASETS="mmlu_cot_zero_shot humaneval cmmlu_zero_shot bbh hellaswag winogrande"
declare -A MODEL_GPU=(
    ["mistral-7b"]="0"
    ["qwen2.5-7b"]="1"
    ["qwen3-8b"]="2"
    ["tower-9b"]="3"
)

timestamp=$(date '+%Y%m%d_%H%M%S')
out_dir="outputs/benchmark_${timestamp}"
mkdir -p "logs" "$out_dir"

for model_name in "${!MODEL_GPU[@]}"; do
    gpu_id="${MODEL_GPU[$model_name]}"
    model_path="$MODEL_BASE/$model_name"
    
    echo "启动模型: $model_name 在 GPU $gpu_id"
    
    tmux new-session -d -s "${model_name}" "
        # 设置环境变量
        export COMPASS_DATA_CACHE=/xfr_ceph_sh/liuchonghan/opencompass_lao
        export OPENCOMPASS_DATASETS_PATH=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export OPENCOMPASS_CACHE_DIR=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export COMPASS_ALLOW_DOWNLOAD=0
        export HF_ENDPOINT=https://hf-mirror.com
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        # 运行评测 - 使用 demo 中的参数格式
        python run.py \\
            --datasets $DATASETS \\
            --hf-type chat \\
            --hf-path '$model_path' \\
            --work-dir '$out_dir/${model_name}' \\
            --max-num-workers 8 \\
            --max-workers-per-gpu 1 \\
            --debug \\
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
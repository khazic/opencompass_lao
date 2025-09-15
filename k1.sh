#!/bin/bash

MODEL_BASE="/xfr_ceph_sh/liuchonghan/paper_model"

DATASETS="mmlu_cot_zero_shot humaneval cmmlu_zero_shot bbh hellaswag winogrande"
declare -A MODEL_GPU=(
    ["RLer_MtPO_allenai_025"]="0"
    ["apertus-8b"]="1"
    ["aya-expanse-8b"]="2"
    ["depsk-7b"]="3"
    ["emma-500"]="4"
    ["gamme3-27b"]="5"
    ["llama3.1-8b"]="6"
    ["llamax3-8b"]="7"
)

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
        export COMPASS_DATA_CACHE=/xfr_ceph_sh/liuchonghan/opencompass_lao
        export OPENCOMPASS_DATASETS_PATH=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export OPENCOMPASS_CACHE_DIR=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
        export COMPASS_ALLOW_DOWNLOAD=0
        export HF_ENDPOINT=https://hf-mirror.com
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        # 运行评测
        python run.py \\
            --models hf_${model_name} \\
            --datasets $DATASETS \\
            --work-dir '$out_dir/${model_name}' \\
            --max-num-workers 8 \\
            --max-workers-per-gpu 1 \\
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
#!/bin/bash

# 设置环境变量以使用本地数据集
export COMPASS_DATA_CACHE=/xfr_ceph_sh/liuchonghan/opencompass_lao
export OPENCOMPASS_DATASETS_PATH=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
export OPENCOMPASS_CACHE_DIR=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
export COMPASS_ALLOW_DOWNLOAD=0
export HF_ENDPOINT=https://hf-mirror.com

# 定义要使用的GPU
GPU_LIST=(0 1 2 3 4 5 6 7)
timestamp=$(date '+%Y%m%d_%H%M%S')
out_dir="outputs/benchmark_${timestamp}"
mkdir -p logs "$out_dir"

# 定义模型列表和对应路径
declare -A models=(
    ["RLer_MtPO_allenai_025"]="/xfr_ceph_sh/liuchonghan/paper_model/RLer_MtPO_allenai_025"
    ["depsk-7b"]="/xfr_ceph_sh/liuchonghan/paper_model/depsk-7b"
    ["mistral-7b"]="/xfr_ceph_sh/liuchonghan/paper_model/mistral-7b"
    ["hunyuan-mt"]="/xfr_ceph_sh/liuchonghan/paper_model/hunyuan-mt"
    ["llama3.1-8b"]="/xfr_ceph_sh/liuchonghan/paper_model/llama3.1-8b"
    ["qwen2.5-7b"]="/xfr_ceph_sh/liuchonghan/paper_model/qwen2.5-7b"
    ["qwen3-8b"]="/xfr_ceph_sh/liuchonghan/paper_model/qwen3-8b"
    ["seedx-mt"]="/xfr_ceph_sh/liuchonghan/paper_model/seedx-mt"
)

# 定义要使用的数据集（选择data目录中已有的常见评测集）
datasets=(
    "mmlu_cot_zero_shot"  # 替换 "mmlu"
    "gsm8k_python"        # 替换 "gsm8k"
    "humaneval_python"    # 替换 "humaneval"
    "cmmlu_zero_shot"     # 替换 "cmmlu"
    "bbh_cot"             # 替换 "bbh"
    "hellaswag"           # 保持不变
    "winogrande"          # 保持不变
)

echo "开始评测..."
echo "使用数据集: ${datasets[*]}"
echo "评测模型数: ${#models[@]}"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 并行评测每个模型
declare -a pids=()
declare -A pid2name=()
slot_idx=0
total_models=${#models[@]}
current_model=0

for model_name in "${!models[@]}"; do
    current_model=$((current_model + 1))
    model_path=${models[$model_name]}
    log_file="logs/${model_name}_${timestamp}.log"
    gpu_id=${GPU_LIST[$(( slot_idx % ${#GPU_LIST[@]} ))]}
    
    echo ""
    echo "[$current_model/$total_models] 并行启动：$model_name -> GPU $gpu_id"
    echo "模型路径：$model_path"
    echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
    
    # 构建数据集参数
    datasets_args=""
    for ds in "${datasets[@]}"; do
        datasets_args+=" $ds"
    done
    
    # 定义要使用的配置文件
    config_file="examples/eval_base_demo.py"

    # 启动评测进程
    (
        CUDA_VISIBLE_DEVICES=$gpu_id \
        HF_ENDPOINT=https://hf-mirror.com \
        OC_HF_TYPE=chat \
        OC_HF_PATH="$model_path" \
        OC_MODEL_ABBR="$model_name" \
        OC_HF_NUM_GPUS=1 \
        OC_BATCH_SIZE=16 \
        OC_MAX_SEQ_LEN=4096 \
        OC_MAX_OUT_LEN=1024 \
        OC_MODEL_KWARGS='{"trust_remote_code": true, "torch_dtype": "torch.bfloat16", "device_map": "auto"}' \
        OC_GENERATION_KWARGS='{"do_sample": false, "num_beams": 1}' \
        opencompass \
            "$config_file" \
            --max-num-workers 8 \
            --max-workers-per-gpu 1 \
            --work-dir "$out_dir/$model_name" \
            --retry 3 \
            --debug \
            > "$log_file" 2>&1
    ) &
    
    pid=$!
    pids+=("$pid")
    pid2name["$pid"]="$model_name"
    slot_idx=$((slot_idx + 1))
done

# 等待所有评测完成
for p in "${pids[@]}"; do
    if wait "$p"; then
        echo "✅ ${pid2name[$p]} 完成 - $(date '+%H:%M:%S')"
    else
        echo "❌ ${pid2name[$p]} 失败 - $(date '+%H:%M:%S')"
        echo "查看错误日志：logs/${pid2name[$p]}_${timestamp}.log"
    fi
done

echo ""
echo "评测完成！"
echo "时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录：$out_dir/"
echo "日志目录：logs/"
#!/bin/bash

export COMPASS_DATA_CACHE=/xfr_ceph_sh/liuchonghan/opencompass_lao
export OPENCOMPASS_DATASETS_PATH=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
export OPENCOMPASS_CACHE_DIR=/xfr_ceph_sh/liuchonghan/opencompass_lao/data
export COMPASS_ALLOW_DOWNLOAD=0
export HF_ENDPOINT=https://hf-mirror.com

GPU_LIST=(0 1 2 3 4 5 6 7)
timestamp=$(date '+%Y%m%d_%H%M%S')
out_dir="outputs/benchmark_${timestamp}"
mkdir -p logs "$out_dir"

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

datasets=(
    "mmlu_cot_zero_shot"  # 替换 "mmlu"
    "gsm8k_python"        # 替换 "gsm8k"
    "humaneval_python"    # 替换 "humaneval"
    "cmmlu_zero_shot"     # 替换 "cmmlu"
    "bbh_cot"             # 替换 "bbh"
    "hellaswag"           # 保持不变
    "winogrande"          # 保持不变
)

# 进度条函数
progress_bar() {
    local width=50
    local percent=$1
    local completed=$((width * percent / 100))
    local remaining=$((width - completed))
    
    printf "["
    printf "%${completed}s" | tr ' ' '#'
    printf "%${remaining}s" | tr ' ' ' '
    printf "] %3d%%\n" "$percent"
}

# 创建进度状态文件
progress_dir="$out_dir/progress"
mkdir -p "$progress_dir"

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

# 启动进度监控函数
monitor_progress() {
    local interval=5  # 更新间隔（秒）
    
    while true; do
        clear
        echo "OpenCompass 评测进度 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        
        # 计算总体进度
        local total_percent=0
        local completed_models=0
        local active_models=0
        
        for model_name in "${!models[@]}"; do
            local progress_file="$progress_dir/${model_name}_progress.txt"
            local status_file="$progress_dir/${model_name}_status.txt"
            local task_file="$progress_dir/${model_name}_task.txt"
            
            if [[ -f "$status_file" ]]; then
                local status=$(cat "$status_file")
                if [[ "$status" == "completed" ]]; then
                    ((completed_models++))
                    total_percent=$((total_percent + 100))
                    echo -e "✅ \033[1;32m${model_name}\033[0m [完成]"
                    progress_bar 100
                elif [[ "$status" == "failed" ]]; then
                    ((completed_models++))
                    total_percent=$((total_percent + 100))
                    echo -e "❌ \033[1;31m${model_name}\033[0m [失败]"
                    progress_bar 100
                else
                    ((active_models++))
                    if [[ -f "$progress_file" ]]; then
                        local percent=$(cat "$progress_file")
                        local current_task=""
                        if [[ -f "$task_file" ]]; then
                            current_task=$(cat "$task_file")
                        fi
                        total_percent=$((total_percent + percent))
                        echo -e "🔄 \033[1;33m${model_name}\033[0m [运行中 - $current_task]"
                        progress_bar "$percent"
                    else
                        echo -e "⏳ \033[1;34m${model_name}\033[0m [等待中]"
                        progress_bar 0
                    fi
                fi
            else
                echo -e "⏳ \033[1;34m${model_name}\033[0m [等待中]"
                progress_bar 0
            fi
        done
        
        # 计算平均进度
        if [[ $total_models -gt 0 ]]; then
            local avg_percent=$((total_percent / total_models))
            echo ""
            echo -e "\033[1;36m总体进度\033[0m: $completed_models/$total_models 模型完成"
            progress_bar "$avg_percent"
        fi
        
        echo ""
        echo "结果目录: $out_dir/"
        echo "日志目录: logs/"
        
        # 如果所有模型都完成了，退出监控
        if [[ $completed_models -eq $total_models ]]; then
            break
        fi
        
        sleep $interval
    done
}

# 启动进度监控（后台运行）
monitor_progress &
monitor_pid=$!

for model_name in "${!models[@]}"; do
    current_model=$((current_model + 1))
    model_path=${models[$model_name]}
    log_file="logs/${model_name}_${timestamp}.log"
    gpu_id=${GPU_LIST[$(( slot_idx % ${#GPU_LIST[@]} ))]}
    
    echo ""
    echo "[$current_model/$total_models] 并行启动：$model_name -> GPU $gpu_id"
    echo "模型路径：$model_path"
    echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
    
    # 定义要使用的配置文件（使用 env 驱动 + 自定义数据集组合）
    config_file="configs/benches/custom_reasoning_python.py"

    # 在第158-159行，不需要构建逗号分隔的字符串
    # 删除这两行:
    # datasets_str=$(IFS=,; echo "${datasets[*]}")

    # 修改第179-181行的参数传递方式
    # 启动评测进程
    (
        # 创建初始进度
        echo "0" > "$progress_dir/${model_name}_progress.txt"
        echo "初始化中..." > "$progress_dir/${model_name}_task.txt"
        
        CUDA_VISIBLE_DEVICES=$gpu_id \
        HF_ENDPOINT=https://hf-mirror.com \
        OC_HF_TYPE=chat \
        OC_HF_PATH="$model_path" \
        OC_MODEL_ABBR="hf_$model_name" \
        OC_HF_NUM_GPUS=1 \
        OC_BATCH_SIZE=16 \
        OC_MAX_SEQ_LEN=4096 \
        OC_MAX_OUT_LEN=1024 \
        OC_MODEL_KWARGS='{"trust_remote_code": true, "dtype": "bfloat16", "device_map": "auto", "attn_implementation": "eager"}' \
        OC_GENERATION_KWARGS='{"do_sample": false, "num_beams": 1, "use_cache": false}' \
        opencompass \
            "$config_file" \
            --max-num-workers 8 \
            --max-workers-per-gpu 1 \
            --work-dir "$out_dir/$model_name" \
            --retry 3 \
            --debug \
            > "$log_file" 2>&1
            
        # 更新进度状态（成功）
        if [ $? -eq 0 ]; then
            echo "completed" > "$progress_dir/${model_name}_status.txt"
        else
            echo "failed" > "$progress_dir/${model_name}_status.txt"
        fi
    ) &
    
    # 启动进度更新进程
    (
        # 每5秒更新一次进度
        while true; do
            # 检查评测是否完成
            if [[ -f "$progress_dir/${model_name}_status.txt" ]]; then
                break
            fi
            
            # 从日志中提取进度信息
            if [[ -f "$log_file" ]]; then
                # 从OpenCompass日志中提取当前任务
                current_task=$(grep -a "INFO - Task" "$log_file" | tail -1 | sed 's/.*Task \[\(.*\)\].*/\1/' 2>/dev/null || echo "准备中")
                if [[ -n "$current_task" ]]; then
                    echo "$current_task" > "$progress_dir/${model_name}_task.txt"
                fi
                
                # 从日志中提取已完成的任务数
                total_tasks=8  # 估计的总任务数
                completed_tasks=$(grep -a "INFO - Task" "$log_file" | wc -l)
                
                if [[ $completed_tasks -gt 0 ]]; then
                    # 计算大致百分比进度
                    percent=$((completed_tasks * 100 / total_tasks))
                    if [[ $percent -gt 100 ]]; then
                        percent=100
                    fi
                    echo "$percent" > "$progress_dir/${model_name}_progress.txt"
                    
                    # 额外显示当前任务的详细进度
                    progress_line=$(grep -a "%" "$log_file" | grep -a "█" | tail -1 2>/dev/null)
                    if [[ -n "$progress_line" ]]; then
                        # 提取当前子任务的百分比
                        subtask_percent=$(echo "$progress_line" | grep -o '[0-9]\+%' | head -1 | tr -d '%')
                        if [[ -n "$subtask_percent" && "$subtask_percent" != "100" ]]; then
                            # 当前大任务的进度基数
                            base_percent=$(( (completed_tasks - 1) * 100 / total_tasks ))
                            # 当前子任务的权重
                            weight=$((100 / total_tasks))
                            # 计算总进度：基础进度 + 当前子任务的加权进度
                            current_percent=$((base_percent + subtask_percent * weight / 100))
                            echo "$current_percent" > "$progress_dir/${model_name}_progress.txt"
                        fi
                    fi
                fi
            fi
            
            sleep 5
        done
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

# 结束进度监控
kill $monitor_pid 2>/dev/null

echo ""
echo "评测完成！"
echo "时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录：$out_dir/"
echo "日志目录：logs/"

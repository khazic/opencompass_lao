#!/bin/bash

cd /xfr_ceph_sh/liuchonghan/opencompass_lao
source /xfr_ceph_sh/liuchonghan/env/etc/profile.d/conda.sh
conda activate opencompass

export HF_ENDPOINT=https://hf-mirror.com

# Select benchmark set: ifeval | safety | subjective | custom
BENCH=${BENCH:-ifeval}

case "$BENCH" in
  ifeval)
    config_path="configs/benches/ifeval_5k.py"
    out_dir="outputs/ifeval_5k"
    ;;
  safety)
    config_path="configs/benches/if_plus_safety_5k.py"
    out_dir="outputs/if_plus_safety_5k"
    ;;
  subjective)
    config_path="configs/benches/subjective_if_5k.py"
    out_dir="outputs/subjective_if_5k"
    ;;
  custom)
    config_path="configs/user_bench_5k.py"
    out_dir="outputs/user_bench_5k"
    ;;
  *)
    config_path="configs/benches/if_plus_safety_5k.py"
    out_dir="outputs/if_plus_safety_5k"
    ;;
esac

mkdir -p logs "$out_dir"
timestamp=$(date '+%Y%m%d_%H%M%S')

declare -A models=(
    ["RLer_MtPO_allenai_025"]="/xfr_ceph_sh/liuchonghan/paper_model/RLer_MtPO_allenai_025"
    ["hunyuan-mt"]="/xfr_ceph_sh/liuchonghan/paper_model/hunyuan-mt"
    ["llama3.1-8b"]="/xfr_ceph_sh/liuchonghan/paper_model/llama3.1-8b"
    ["qwen2.5-7b"]="/xfr_ceph_sh/liuchonghan/paper_model/qwen2.5-7b"
    ["qwen3-8b"]="/xfr_ceph_sh/liuchonghan/paper_model/qwen3-8b"
    ["seedx-mt"]="/xfr_ceph_sh/liuchonghan/paper_model/seedx-mt"
)

# Config selected via $BENCH

echo "开始评测..."
echo "评测配置: $config_path (每个评测集≤5000样本)"
echo "Benchmark: $BENCH"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

total_models=${#models[@]}
current_model=0

GPU_LIST=(0 1 2 3 4 5 6 7)

declare -a pids=()
declare -A pid2name=()
max_parallel=${#GPU_LIST[@]}
slot_idx=0

for model_name in "${!models[@]}"; do
    current_model=$((current_model + 1))
    model_path=${models[$model_name]}
    log_file="logs/${model_name}_${timestamp}.log"
    gpu_id=${GPU_LIST[$(( slot_idx % max_parallel ))]}

    echo ""
    echo "[$current_model/$total_models] 并行启动：$model_name -> GPU $gpu_id"
    echo "模型路径：$model_path"
    echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"

    (
      CUDA_VISIBLE_DEVICES=$gpu_id HF_ENDPOINT=https://hf-mirror.com opencompass \
          "$config_path" \
          --hf-type chat \
          --hf-path "$model_path" \
          --max-num-workers 8 \
          --work-dir "$out_dir" \
          --batch-size 16 \
          --hf-num-gpus 1 \
          --max-seq-len 4096 \
          --max-out-len 1024 \
          --max-workers-per-gpu 1 \
          --model-kwargs "trust_remote_code=True dtype=bfloat16 device_map=auto" \
          --generation-kwargs "do_sample=False num_beams=1" \
          --retry 3 \
          --debug \
          > "$log_file" 2>&1
    ) &
    pid=$!
    pids+=("$pid")
    pid2name["$pid"]="$model_name"
    slot_idx=$((slot_idx + 1))

    # 若模型数超过GPU数，分批并行：每满一批就等待一轮
    if (( slot_idx % max_parallel == 0 )); then
        for p in "${pids[@]}"; do
            if wait "$p"; then
                echo "✅ ${pid2name[$p]} 完成 - $(date '+%H:%M:%S')"
            else
                echo "❌ ${pid2name[$p]} 失败 - $(date '+%H:%M:%S')"
                echo "查看错误日志：logs/${pid2name[$p]}_${timestamp}.log"
            fi
        done
        # 清空本批次PID
        pids=()
    fi
done

# 等待剩余后台任务
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

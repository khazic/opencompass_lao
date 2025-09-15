#!/bin/bash

cd /xfr_ceph_sh/liuchonghan/opencompass_lao

# Activate conda env only if not already in it, and do it robustly
if [[ "${CONDA_DEFAULT_ENV:-}" != "opencompass" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate opencompass || true
  else
    echo "[warn] conda not found in PATH; assuming env already active" >&2
  fi
fi

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

# 统计本次将运行的评测集数量
ds_count=$(python - "$config_path" <<'PY'
import sys
from mmengine.config import Config
cfg = Config.fromfile(sys.argv[1])
print(len(cfg.get('datasets', [])))
PY
)
echo "本次评测集总数：$ds_count"

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
      CUDA_VISIBLE_DEVICES=$gpu_id \
      HF_ENDPOINT=https://hf-mirror.com \
      OC_HF_TYPE=chat \
      OC_HF_PATH="$model_path" \
      OC_MODEL_ABBR="$model_name" \
      OC_HF_NUM_GPUS=1 \
      OC_BATCH_SIZE=16 \
      OC_MAX_SEQ_LEN=4096 \
      OC_MAX_OUT_LEN=1024 \
      OC_MODEL_KWARGS='{"trust_remote_code": true, "dtype": "bfloat16", "device_map": "auto"}' \
      OC_GENERATION_KWARGS='{"do_sample": false, "num_beams": 1}' \
      opencompass \
          "$config_path" \
          --max-num-workers 8 \
          --max-workers-per-gpu 1 \
          --work-dir "$out_dir" \
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

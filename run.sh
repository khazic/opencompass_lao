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

# Proxy normalization and logging (mask credentials in output)
mask_url() {
  local u="$1"
  if [[ -z "$u" ]]; then echo "none"; return; fi
  echo "$u" | sed -E 's#(://)[^/@]+@#\1***:***@#'
}

# Sync upper/lower case envs so Python/curl both see them
if [[ -n "${http_proxy:-}" && -z "${HTTP_PROXY:-}" ]]; then export HTTP_PROXY="$http_proxy"; fi
if [[ -n "${https_proxy:-}" && -z "${HTTPS_PROXY:-}" ]]; then export HTTPS_PROXY="$https_proxy"; fi
if [[ -z "${ALL_PROXY:-}" && -n "${https_proxy:-}" ]]; then export ALL_PROXY="$https_proxy"; fi

echo "Proxy: http=$(mask_url "${http_proxy:-}") https=$(mask_url "${https_proxy:-}") all=$(mask_url "${ALL_PROXY:-}") no_proxy=${no_proxy:-none}"

# Prefer local dataset cache to avoid downloads in restricted envs
export COMPASS_DATA_CACHE=${COMPASS_DATA_CACHE:-/xfr_ceph_sh/liuchonghan/opencompass_data}
mkdir -p "$COMPASS_DATA_CACHE"

# Select benchmark set: ifeval | safety | subjective | custom
BENCH=${BENCH:-ifeval}

case "$BENCH" in
  ifeval)
    config_path="configs/benches/ifeval_5k.py"
    out_dir="outputs/ifeval_5k"
    required_data=("data/ifeval")
    ;;
  safety)
    config_path="configs/benches/if_plus_safety_5k.py"
    out_dir="outputs/if_plus_safety_5k"
    required_data=("data/ifeval" "data/civilcomments" "data/crowspairs" "data/cvalues")
    ;;
  subjective)
    config_path="configs/benches/subjective_if_5k.py"
    out_dir="outputs/subjective_if_5k"
    required_data=("data/ifeval")
    ;;
  custom)
    config_path="configs/user_bench_5k.py"
    out_dir="outputs/user_bench_5k"
    required_data=( )
    ;;
  *)
    config_path="configs/benches/if_plus_safety_5k.py"
    out_dir="outputs/if_plus_safety_5k"
    required_data=("data/ifeval" "data/civilcomments" "data/crowspairs" "data/cvalues")
    ;;
esac

mkdir -p logs "$out_dir"
timestamp=$(date '+%Y%m%d_%H%M%S')

# Prefetch critical local-mode datasets via proxy if available (avoids 503)
prefetch_datasets() {
  local base="$COMPASS_DATA_CACHE/data"
  mkdir -p "$base"
  # IFEval expects: $COMPASS_DATA_CACHE/data/ifeval/input_data.jsonl
  if [[ ! -s "$base/ifeval/input_data.jsonl" ]]; then
    echo "[prefetch] IFEval 本地未找到，尝试通过代理预取..."
    # Prefer HTTPS; some proxies屏蔽明文HTTP
    local url="https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip"
    local z="$base/ifeval.zip"
    # Use curl if available; it respects http_proxy/https_proxy
    if command -v curl >/dev/null 2>&1; then
      for i in 1 2 3; do
        echo "[prefetch] 下载 IFEval (重试第 $i 次)..."
        if curl -L --fail --retry 3 --retry-delay 2 \
              --connect-timeout 15 --max-time 300 \
              -o "$z" "$url"; then
          break
        fi
        sleep 2
      done
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$z" "$url" || true
    fi
    if [[ -s "$z" ]]; then
      echo "[prefetch] 解压 IFEval..."
      mkdir -p "$base/ifeval"
      if command -v unzip >/dev/null 2>&1; then
        unzip -o "$z" -d "$base/ifeval" >/dev/null 2>&1 || true
      else
        python - "$z" "$base/ifeval" <<'PY'
import sys, zipfile, os
z, dst = sys.argv[1:]
os.makedirs(dst, exist_ok=True)
with zipfile.ZipFile(z, 'r') as f:
    f.extractall(dst)
PY
      fi
      if [[ -s "$base/ifeval/input_data.jsonl" ]]; then
        echo "[prefetch] IFEval 预取完成。"
      else
        echo "[prefetch] 解压后未找到 input_data.jsonl，将回退为运行时自动下载。"
      fi
    else
      echo "[prefetch] 下载失败，将回退为运行时自动下载。"
    fi
  fi
}

prefetch_datasets

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

# Config selected via $BENCH

echo "开始评测..."
echo "评测配置: $config_path (每个评测集≤5000样本)"
echo "Benchmark: $BENCH"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Preflight: check required datasets exist in cache.
# If proxies are set or COMPASS_ALLOW_DOWNLOAD=1, allow auto-download; else fail fast.
allow_download=${COMPASS_ALLOW_DOWNLOAD:-1}
if [[ -n "${http_proxy:-}${https_proxy:-}" ]]; then
  allow_download=1
fi
if [[ "$allow_download" != "1" ]]; then
  missing=()
  for d in "${required_data[@]}"; do
    if [[ ! -e "$COMPASS_DATA_CACHE/$d" ]]; then
      missing+=("$d")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "[error] 缺少本地数据集缓存，且未检测到代理或显式允许下载 (COMPASS_ALLOW_DOWNLOAD=1)。" >&2
    echo "需要存在于 \"$COMPASS_DATA_CACHE\" 下的目录：" >&2
    for d in "${missing[@]}"; do echo "  - $d" >&2; done
    echo "解决方案：" >&2
    echo "1) 在有网络的机器下载 opencompass 数据集压缩包并解压到 $COMPASS_DATA_CACHE/ 对应目录；或" >&2
    echo "2) 设置代理变量后再运行（已支持 http_proxy/https_proxy）；或" >&2
    echo "3) 设定 COMPASS_ALLOW_DOWNLOAD=1 允许自动下载；或" >&2
    echo "4) 修改 COMPASS_DATA_CACHE 指向已有数据集的共享路径。" >&2
    exit 1
  fi
fi

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
      OC_MODEL_KWARGS='{"trust_remote_code": true, "torch_dtype": "torch.bfloat16", "device_map": "auto"}' \
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

#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}�� OpenCompass Qwen2 模型评测一键启动脚本${NC}"
echo "========================================"

# 关闭代理
echo -e "${YELLOW}🔧 关闭代理...${NC}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
echo -e "${GREEN}✅ 代理已关闭${NC}"

# 设置工作目录和环境
cd /xfr_ceph_sh/liuchonghan/opencompass_lao

# 激活conda环境
source /llm-align/liuchonghan/env/etc/profile.d/conda.sh
conda activate opencompass

# 定义模型路径和名称
declare -A models
models=(
    ["SFT模型"]="/xfr_ceph_sh/liuchonghan/OpenRLHF/examples/scripts/checkpoint/qwen2_5_sft_domain"
    ["7B_INS模型"]="/llm-align/liuchonghan/opensource_models/qwen2_5_7b_ins"
    ["8Langs_CPT模型"]="/llm-align/duyimin/multi_lang/Qwen2.5-7B-8Langs-CPT-250724"
    ["CPT_ALL模型"]="/llm-align/duyimin/duyimin/tools/Qwen/trans_7b_cpt_all_data/checkpoint-9000"
)

# 定义测试配置 - 修改为包含更多数据集
declare -A test_configs
test_configs=(
    ["1"]="humaneval_gen|代码能力快速测试"
    ["2"]="humaneval_gen mmlu_gen|代码+知识标准测试"
    ["3"]="humaneval_gen mmlu_gen math_gen|代码+知识+数学完整测试"
    ["4"]="humaneval_gen mmlu_gen math_gen commonsenseqa_gen|全面能力评测"
    ["5"]="math_gen|数学推理专项测试"
    ["6"]="mmlu_gen|知识理解专项测试"
    ["7"]="humaneval_gen mmlu_gen math_gen commonsenseqa_gen gsm8k_gen|超全面评测"
    ["8"]="custom|自定义数据集组合"
)

# 创建日志目录
mkdir -p logs
timestamp=$(date '+%Y%m%d_%H%M%S')

echo ""
echo -e "${BLUE}📊 请选择测试配置：${NC}"
echo "----------------------------------------"
for key in $(echo "${!test_configs[@]}" | tr ' ' '\n' | sort -n); do
    IFS='|' read -r datasets description <<< "${test_configs[$key]}"
    echo -e "${PURPLE}$key${NC}. $description"
    if [[ "$key" != "8" ]]; then
        echo "   数据集: ${CYAN}$datasets${NC}"
    fi
done
echo "----------------------------------------"

# 读取用户选择
echo -n -e "${YELLOW}请输入选择 (1-8) [默认: 7]: ${NC}"
read -r choice

# 设置默认选择
if [[ -z "$choice" ]]; then
    choice="7"
fi

# 验证选择
if [[ ! "${test_configs[$choice]}" ]]; then
    echo -e "${RED}❌ 无效选择，使用默认配置 4${NC}"
    choice="7"
fi

# 解析选择的配置
IFS='|' read -r test_datasets test_description <<< "${test_configs[$choice]}"

# 如果是自定义配置，让用户输入数据集
if [[ "$choice" == "8" ]]; then
    echo ""
    echo -e "${CYAN}📝 可用的数据集列表：${NC}"
    echo "humaneval_gen, mmlu_gen, math_gen, commonsenseqa_gen, gsm8k_gen,"
    echo "ceval_gen, cmmlu_gen, bbh_gen, bbeh_gen, bigcodebench_gen"
    echo ""
    echo -n -e "${YELLOW}请输入数据集名称（用空格分隔）: ${NC}"
    read -r custom_datasets
    if [[ -n "$custom_datasets" ]]; then
        test_datasets="$custom_datasets"
        test_description="自定义数据集组合"
    else
        echo -e "${RED}❌ 未输入数据集，使用默认配置 3${NC}"
        choice="3"
        IFS='|' read -r test_datasets test_description <<< "${test_configs[3]}"
    fi
fi

echo ""
echo -e "${GREEN}🎯 已选择配置 $choice: $test_description${NC}"
echo -e "${CYAN}📁 测试数据集: $test_datasets${NC}"
echo ""

# 在nohup模式下默认后台运行
background_run="y"

# 计数器
total_models=${#models[@]}
current_model=0

# 主评测函数
run_evaluation() {
    # 遍历模型并进行评测
    for model_name in "${!models[@]}"; do
        current_model=$((current_model + 1))
        model_path=${models[$model_name]}
        
        echo ""
        echo -e "${BLUE}🔄 [$current_model/$total_models] ========================================${NC}"
        echo -e "${PURPLE}�� 开始测试：$model_name${NC}"
        echo -e "${CYAN}�� 模型路径：$model_path${NC}"
        echo -e "${CYAN}�� 数据集：$test_datasets${NC}"
        echo -e "${YELLOW}⏰ 开始时间：$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo "========================================"

        # 创建模型特定的日志文件
        log_file="logs/${model_name}_${timestamp}.log"
        
        # 执行评测
        echo -e "${YELLOW}⚡ 正在执行评测...${NC}"
        
        # 构建命令
        cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 opencompass --hf-type chat --hf-path \"$model_path\" --datasets $test_datasets --max-num-workers 8 --work-dir outputs/main_bench --batch-size 16 --num-gpus 8 --max-seq-len 2048 --max-out-len 1024 --max-workers-per-gpu 1"
        
        # 执行命令
        if [[ "$background_run" =~ ^[Yy]$ ]]; then
            # 后台运行时同时显示进度
            eval "$cmd" 2>&1 | tee "$log_file" | while IFS= read -r line; do
                if [[ $line == *"progress"* ]] || [[ $line == *"accuracy"* ]] || [[ $line == *"score"* ]] || [[ $line == *"Evaluating"* ]]; then
                    echo -e "${CYAN}[进度] $line${NC}"
                fi
            done
            exit_code=${PIPESTATUS[0]}
        else
            # 前台运行时实时显示所有输出
            eval "$cmd" 2>&1 | tee "$log_file" | while IFS= read -r line; do
                if [[ $line == *"progress"* ]] || [[ $line == *"accuracy"* ]] || [[ $line == *"score"* ]] || [[ $line == *"Evaluating"* ]]; then
                    echo -e "${CYAN}[进度] $line${NC}"
                else
                    echo "$line"
                fi
            done
            exit_code=${PIPESTATUS[0]}
        fi

        # 检查评测结果
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}✅ $model_name 测试完成 - $(date '+%H:%M:%S')${NC}"
            echo -e "${CYAN}📄 日志保存在：$log_file${NC}"
        else
            echo -e "${RED}❌ $model_name 测试失败 - $(date '+%H:%M:%S')${NC}"
            echo -e "${RED}📄 错误日志保存在：$log_file${NC}"
            echo -e "${YELLOW}⚠️  继续测试下一个模型...${NC}"
        fi
        
        # 模型间间隔
        if [ $current_model -lt $total_models ]; then
            echo -e "${YELLOW}⏳ 等待10秒后继续下一个模型...${NC}"
            sleep 10
        fi
    done
}

# 执行评测
if [[ "$background_run" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}�� 后台运行模式启动...${NC}"
    echo -e "${CYAN}📄 主日志文件: logs/main_${timestamp}.log${NC}"
    
    # 后台运行
    {
        run_evaluation
    } > "logs/main_${timestamp}.log" 2>&1 &
    
    bg_pid=$!
    echo -e "${GREEN}✅ 评测已在后台启动 (PID: $bg_pid)${NC}"
    echo ""
    echo -e "${CYAN}🔍 监控命令：${NC}"
    echo "   实时查看主日志: tail -f logs/main_${timestamp}.log"
    echo "   查看进程状态: ps aux | grep $bg_pid"
    echo "   查看输出目录: watch -n 30 'ls -la outputs/main_bench/'"
else
    # 前台运行
    run_evaluation
fi

echo ""
echo -e "${GREEN}🎉 ========================================${NC}"
echo -e "${CYAN}✨ 评测脚本执行完成！${NC}"
echo -e "${PURPLE}📊 测试配置：$test_description${NC}"
echo -e "${CYAN}�� 数据集：$test_datasets${NC}"
echo -e "${YELLOW}⏰ 时间：$(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""
echo -e "${BLUE}�� 结果查看：${NC}"
echo "   - 主要结果目录：outputs/main_bench/"
echo "   - 详细日志目录：logs/"
echo "   - 最新输出：ls -la outputs/main_bench/"
echo ""
echo -e "${CYAN}🔍 快速查看结果命令：${NC}"
echo "   ls -la outputs/main_bench/ | tail -10"
echo "   find outputs/main_bench/ -name '*.json' -newer logs/ 2>/dev/null"
echo "========================================"
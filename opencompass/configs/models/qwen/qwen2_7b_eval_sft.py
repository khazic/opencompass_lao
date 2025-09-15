from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_genericllmeval_academic_gen import aime2024_datasets
    from opencompass.configs.datasets.bbh.bbh_0shot_nocot_academic_gen import bbh_datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets
    from opencompass.configs.datasets.math.math_prm800k_500_gen import math_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets

models = [dict(
    type=HuggingFaceCausalLM,
    path='/xfr_ceph_sh/liuchonghan/OpenRLHF/examples/scripts/checkpoint/qwen2_5_sft_domain',
    tokenizer_path='/xfr_ceph_sh/liuchonghan/OpenRLHF/examples/scripts/checkpoint/qwen2_5_sft_domain',
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True,
        torch_dtype='auto',
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    ),
    max_out_len=32768,
    max_seq_len=32768,
    batch_size=4,
    run_cfg=dict(num_gpus=4)
)]

datasets = (
    aime2024_datasets +
    bbh_datasets +
    gpqa_datasets +
    humaneval_datasets +
    ifeval_datasets +
    math_datasets +
    mmlu_pro_datasets +
    LCB_datasets
)

# LLM judge config
judge_cfg = dict()
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['max_out_len'] = 32768
    if 'judge_cfg' in dataset['eval_cfg']['evaluator']:
        dataset['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['bbh', 'naive_average'],
            ['math_prm800k_500', 'accuracy'],
            ['aime2024', 'accuracy'],
            ['GPQA_diamond', 'accuracy'],
            ['mmlu_pro', 'naive_average'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['lcb_code_generation', 'pass@1'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        ['bbh', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        '',
        'Math Calculation',
        ['math_prm800k_500', 'accuracy'],
        ['aime2024', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['lcb_code_generation', 'pass@1'],
    ],
    summary_groups=core_summary_groups,
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)))

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask)))

work_dir = './outputs/oc_academic_202502_sft'

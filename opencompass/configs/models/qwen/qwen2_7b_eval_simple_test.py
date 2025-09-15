from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    # 只测试一个简单的数据集
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets

# 单个模型配置
models = [dict(
    type=HuggingFaceCausalLM,
    path='/xfr_ceph_sh/liuchonghan/OpenRLHF/examples/scripts/checkpoint/qwen2_5_sft_domain',
    tokenizer_path='/xfr_ceph_sh/liuchonghan/OpenRLHF/examples/scripts/checkpoint/qwen2_5_sft_domain',
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True,
        torch_dtype='auto',
        pad_token_id=151643,  # 明确设置pad_token_id
        attn_implementation='eager',  # 使用eager attention避免兼容性问题
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        pad_token='<|endoftext|>',  # 明确设置pad_token
    ),
    max_out_len=512,   # 进一步减小输出长度
    max_seq_len=2048,  # 进一步减小序列长度
    batch_size=1,      # 最小batch_size
    run_cfg=dict(num_gpus=1),  # 只用1个GPU
    generation_kwargs=dict(
        do_sample=False,
        temperature=1.0,
        pad_token_id=151643,
        eos_token_id=151643,
        use_cache=True,
    )
)]

# 只用一个数据集
datasets = humaneval_datasets

# 简化推理配置
infer = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask)))

# 简化评估配置  
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLEvalTask)))

work_dir = './outputs/simple_test'

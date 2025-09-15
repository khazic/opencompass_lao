import os
from mmengine.config import read_base
from opencompass.models import OpenAISDK  # for judge model cfg
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import SubjectiveEvalTask

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.followbench.followbench_llmeval import followbench_llmeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.fofo.fofo_bilingual_judge_new import fofo_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm_new import alignbench_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4_new import alpacav2_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare_new import arenahard_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import wildbench_datasets  # noqa: F401, F403


def _cap(ds_list, test_cap='[:5000]'):
    capped = []
    for ds in ds_list:
        cfg = dict(ds)
        reader_cfg = dict(cfg.get('reader_cfg', {}))
        reader_cfg['test_range'] = test_cap
        cfg['reader_cfg'] = reader_cfg
        capped.append(cfg)
    return capped


def _build_judge_model_from_env():
    api_key = os.environ.get('OC_JUDGE_API_KEY')
    model = os.environ.get('OC_JUDGE_MODEL', 'gpt-4o-2024-05-13')
    api_base = os.environ.get('OC_JUDGE_API_BASE', 'https://api.openai.com/v1/')
    if not api_key:
        return None
    return dict(
        type=OpenAISDK,
        path=model,
        key=api_key,
        openai_api_base=[api_base],
        batch_size=256,
        temperature=0.0,
        tokenizer_path=model,
        max_out_len=8192,
        max_seq_len=49152,
        verbose=False,
    )


datasets = []
# Always include IFEval (objective IF)
datasets += _cap(ifeval_datasets)

# Subjective IF/format/alignment sets (enabled only when judge key is provided)
_judge_model = _build_judge_model_from_env()
if _judge_model is not None:
    datasets += _cap(followbench_llmeval_datasets)
    datasets += _cap(fofo_datasets)
    datasets += _cap(alignbench_datasets)
    datasets += _cap(alpacav2_datasets)
    datasets += _cap(arenahard_datasets)
    datasets += _cap(wildbench_datasets)

    # Configure subjective evaluation task with LLM judge
    eval = dict(
        partitioner=dict(
            type=SubjectiveNaivePartitioner,
            judge_models=[_judge_model],
        ),
        runner=dict(
            type=LocalRunner,
            max_num_workers=8,
            task=dict(type=SubjectiveEvalTask)
        ),
    )

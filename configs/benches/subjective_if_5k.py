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


datasets = []

# Always include IFEval (objective IF), capped to 5k
for _d in ifeval_datasets:
    _cfg = dict(_d)
    _reader_cfg = dict(_cfg.get('reader_cfg', {}))
    _reader_cfg['test_range'] = '[:5000]'
    _cfg['reader_cfg'] = _reader_cfg
    datasets.append(_cfg)

# Subjective IF/format/alignment sets (enabled only when judge key is provided)
_oc_key = os.environ.get('OC_JUDGE_API_KEY')
_oc_model = os.environ.get('OC_JUDGE_MODEL', 'gpt-4o-2024-05-13')
_oc_base = os.environ.get('OC_JUDGE_API_BASE', 'https://api.openai.com/v1/')

if _oc_key:
    for _src in (
        followbench_llmeval_datasets,
        fofo_datasets,
        alignbench_datasets,
        alpacav2_datasets,
        arenahard_datasets,
        wildbench_datasets,
    ):
        for _d in _src:
            _cfg = dict(_d)
            _reader_cfg = dict(_cfg.get('reader_cfg', {}))
            _reader_cfg['test_range'] = '[:5000]'
            _cfg['reader_cfg'] = _reader_cfg
            datasets.append(_cfg)

    # Configure subjective evaluation task with LLM judge
    eval = dict(
        partitioner=dict(
            type=SubjectiveNaivePartitioner,
            judge_models=[dict(
                type=OpenAISDK,
                path=_oc_model,
                key=_oc_key,
                openai_api_base=[_oc_base],
                batch_size=256,
                temperature=0.0,
                tokenizer_path=_oc_model,
                max_out_len=8192,
                max_seq_len=49152,
                verbose=False,
            )],
        ),
        runner=dict(
            type=LocalRunner,
            max_num_workers=8,
            task=dict(type=SubjectiveEvalTask)
        ),
    )

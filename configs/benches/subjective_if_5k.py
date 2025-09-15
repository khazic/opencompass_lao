import os
import json
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

# Build a single HF model from environment variables
from opencompass.models import (  # noqa: E402
    HuggingFaceBaseModel,
    HuggingFacewithChatTemplate,
)

def _get_env_json(key: str, default: str):
    try:
        return json.loads(os.environ.get(key, default) or default)
    except Exception:
        return json.loads(default)

_hf_path = os.environ.get('OC_HF_PATH', '')
_abbr = os.environ.get('OC_MODEL_ABBR', os.path.basename(_hf_path.rstrip('/')) or 'hf_model')
_hf_type = os.environ.get('OC_HF_TYPE', 'chat')

_num_gpus = int(os.environ.get('OC_HF_NUM_GPUS', '1'))
_batch_size = int(os.environ.get('OC_BATCH_SIZE', '16'))
_max_seq_len = int(os.environ.get('OC_MAX_SEQ_LEN', '4096'))
_max_out_len = int(os.environ.get('OC_MAX_OUT_LEN', '1024'))

_model_kwargs = _get_env_json('OC_MODEL_KWARGS', '{"trust_remote_code": true, "dtype": "bfloat16", "device_map": "auto"}')
_generation_kwargs = _get_env_json('OC_GENERATION_KWARGS', '{"do_sample": false, "num_beams": 1}')
_tokenizer_path = os.environ.get('OC_TOKENIZER_PATH', None)
_tokenizer_kwargs = _get_env_json('OC_TOKENIZER_KWARGS', '{}')
_peft_path = os.environ.get('OC_PEFT_PATH', None)
_peft_kwargs = _get_env_json('OC_PEFT_KWARGS', '{}')
_pad_token_id = os.environ.get('OC_PAD_TOKEN_ID', None)
_stop_words = _get_env_json('OC_STOP_WORDS', '[]')

_mod = HuggingFacewithChatTemplate if _hf_type == 'chat' else HuggingFaceBaseModel

models = [
    dict(
        type=_mod,
        abbr=_abbr,
        path=_hf_path,
        model_kwargs=_model_kwargs,
        tokenizer_path=_tokenizer_path,
        tokenizer_kwargs=_tokenizer_kwargs,
        peft_path=_peft_path,
        peft_kwargs=_peft_kwargs,
        generation_kwargs=_generation_kwargs,
        max_seq_len=_max_seq_len,
        max_out_len=_max_out_len,
        batch_size=_batch_size,
        pad_token_id=None if _pad_token_id is None else int(_pad_token_id),
        stop_words=_stop_words,
        run_cfg=dict(num_gpus=_num_gpus),
    )
]

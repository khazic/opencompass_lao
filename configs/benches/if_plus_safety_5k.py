import os
import json
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.civilcomments.civilcomments_clp import civilcomments_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.crowspairs.crowspairs_gen import crowspairs_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.cvalues.cvalues_responsibility_gen import cvalues_datasets  # noqa: F401, F403


datasets = []
for _src in (ifeval_datasets, civilcomments_datasets, crowspairs_datasets, cvalues_datasets):
    for _d in _src:
        _cfg = dict(_d)
        _reader_cfg = dict(_cfg.get('reader_cfg', {}))
        _reader_cfg['test_range'] = '[:5000]'
        _cfg['reader_cfg'] = _reader_cfg
        datasets.append(_cfg)

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

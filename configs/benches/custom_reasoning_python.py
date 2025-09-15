import os
import json
from mmengine.config import read_base

# Compose the requested datasets:
# - mmlu_cot_zero_shot -> mmlu/mmlu_zero_shot_gen_47e2c0
# - gsm8k_python       -> gsm8k/gsm8k_agent_gen_c3dff3 (Python tool use)
# - humaneval_python   -> humaneval/humaneval_gen_66a7f4
# - cmmlu_zero_shot    -> cmmlu/cmmlu_0shot_cot_gen_305931 (0-shot CoT)
# - bbh_cot            -> bbh/bbh_gen (CoT-style prompts)
# - hellaswag          -> hellaswag/hellaswag_gen_6faab5
# - winogrande         -> winogrande/winogrande_gen_458220

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_zero_shot_gen_47e2c0 import mmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_agent_gen_c3dff3 import gsm8k_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.humaneval.humaneval_gen_66a7f4 import humaneval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.winogrande.winogrande_gen_458220 import winogrande_datasets  # noqa: F401, F403


datasets = []
for _src in (
    mmlu_datasets,
    gsm8k_datasets,
    humaneval_datasets,
    cmmlu_datasets,
    bbh_datasets,
    hellaswag_datasets,
    winogrande_datasets,
):
    datasets.extend(list(_src))

# Build a single HF model from environment variables
from opencompass.models import (  # noqa: E402
    HuggingFaceBaseModel,
    HuggingFacewithChatTemplate,
)

_hf_path = os.environ.get('OC_HF_PATH', '')
_abbr = os.environ.get('OC_MODEL_ABBR', os.path.basename(_hf_path.rstrip('/')) or 'hf_model')
_hf_type = os.environ.get('OC_HF_TYPE', 'chat')

_num_gpus = int(os.environ.get('OC_HF_NUM_GPUS', '1'))
_batch_size = int(os.environ.get('OC_BATCH_SIZE', '16'))
_max_seq_len = int(os.environ.get('OC_MAX_SEQ_LEN', '4096'))
_max_out_len = int(os.environ.get('OC_MAX_OUT_LEN', '1024'))

try:
    _model_kwargs = json.loads(os.environ.get('OC_MODEL_KWARGS', '{"trust_remote_code": true, "dtype": "bfloat16", "device_map": "auto"}'))
except Exception:
    _model_kwargs = {"trust_remote_code": True, "dtype": "bfloat16", "device_map": "auto"}
try:
    _generation_kwargs = json.loads(os.environ.get('OC_GENERATION_KWARGS', '{"do_sample": false, "num_beams": 1}'))
except Exception:
    _generation_kwargs = {"do_sample": False, "num_beams": 1}
_tokenizer_path = os.environ.get('OC_TOKENIZER_PATH', None)
try:
    _tokenizer_kwargs = json.loads(os.environ.get('OC_TOKENIZER_KWARGS', '{}'))
except Exception:
    _tokenizer_kwargs = {}
_peft_path = os.environ.get('OC_PEFT_PATH', None)
try:
    _peft_kwargs = json.loads(os.environ.get('OC_PEFT_KWARGS', '{}'))
except Exception:
    _peft_kwargs = {}
_pad_token_id = os.environ.get('OC_PAD_TOKEN_ID', None)
try:
    _stop_words = json.loads(os.environ.get('OC_STOP_WORDS', '[]'))
except Exception:
    _stop_words = []

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

# Avoid dumping module objects in generated config files
try:
    del json
except Exception:
    pass
try:
    del os
except Exception:
    pass


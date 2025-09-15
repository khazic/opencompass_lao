from mmengine.config import read_base

# Compose multiple built-in datasets and cap test size to 5k
with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets  # noqa: F401, F403


datasets = []
for _src in (ifeval_datasets, mmlu_datasets, bbh_datasets, gsm8k_datasets, ceval_datasets):
    for _d in _src:
        _cfg = dict(_d)
        _reader_cfg = dict(_cfg.get('reader_cfg', {}))
        _reader_cfg['test_range'] = '[:5000]'
        _cfg['reader_cfg'] = _reader_cfg
        datasets.append(_cfg)

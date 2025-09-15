from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403


datasets = []
for _d in ifeval_datasets:
    _cfg = dict(_d)
    _reader_cfg = dict(_cfg.get('reader_cfg', {}))
    _reader_cfg['test_range'] = '[:5000]'
    _cfg['reader_cfg'] = _reader_cfg
    datasets.append(_cfg)

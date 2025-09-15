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

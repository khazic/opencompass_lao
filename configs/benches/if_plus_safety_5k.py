from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.civilcomments.civilcomments_clp import civilcomments_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.crowspairs.crowspairs_gen import crowspairs_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.cvalues.cvalues_responsibility_gen import cvalues_datasets  # noqa: F401, F403


def _cap(ds_list, test_cap='[:5000]'):
    capped = []
    for ds in ds_list:
        cfg = dict(ds)
        reader_cfg = dict(cfg.get('reader_cfg', {}))
        reader_cfg['test_range'] = test_cap
        cfg['reader_cfg'] = reader_cfg
        capped.append(cfg)
    return capped


datasets = []
datasets += _cap(ifeval_datasets)
datasets += _cap(civilcomments_datasets)
datasets += _cap(crowspairs_datasets)
datasets += _cap(cvalues_datasets)


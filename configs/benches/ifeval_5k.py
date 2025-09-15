from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403


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


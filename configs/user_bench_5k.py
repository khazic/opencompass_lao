from mmengine.config import read_base

# Compose multiple built-in datasets and cap test size to 5k

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets  # noqa: F401, F403


def _cap_reader_cfg(ds_list, test_cap='[:5000]'):
    capped = []
    for ds in ds_list:
        cfg = dict(ds)
        reader_cfg = dict(cfg.get('reader_cfg', {}))
        # Inject test_range to limit evaluation samples
        reader_cfg['test_range'] = test_cap
        cfg['reader_cfg'] = reader_cfg
        capped.append(cfg)
    return capped


# Merge datasets and apply 5k cap per dataset
datasets = []
datasets += _cap_reader_cfg(ifeval_datasets)
datasets += _cap_reader_cfg(mmlu_datasets)
datasets += _cap_reader_cfg(bbh_datasets)
datasets += _cap_reader_cfg(gsm8k_datasets)
datasets += _cap_reader_cfg(ceval_datasets)


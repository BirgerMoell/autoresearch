# Russian Transfer Experiment Reproducer

This directory contains the reproducer for the Russian inclusion transfer
experiment reported in `docs/russian-transfer-experiment`.

The core script is:

```text
mix_matched_russian_experiment.py
```

It parses the production-like language allocation table embedded in the script,
collapses language-coded rows into sampling weights, then compares:

- `base_no_ru`
- `with_ru_same_total`
- `with_ru_extra_budget`

The script expects a local directory of language parquet shards, with one
subdirectory per local language code. In the GPU run, these were one-shard
FinePDFs-Edu samples under:

```text
/home/ubuntu/birger/russian-transfer-experiment-data/raw
```

The result artifacts from the reference run are copied into:

```text
docs/russian-transfer-experiment/
```

Use the report page for interpretation and caveats.


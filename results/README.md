# Results

This public repository focuses on the LAMP-Merge implementation and execution flow. Full comparison tables are reported in the paper and are not bundled here because this release intentionally does not include comparison baseline implementations.

The standard evaluation output produced by `evaluate.py` is written to:

```text
<output_root>/eval/<task_type>/<dataset>/<model>/clients_<K>/beta_<b>/seed_<s>/lamp_merge/eval.json
```

Batch runs with `scripts/run_lamp_merge_eval.py` additionally write:

```text
<output_root>/lamp_merge_status.csv
<output_root>/reports/eval_summary.csv
<output_root>/reports/merge_summary.csv
```

## Released Analysis Outputs

The strict medical-versus-natural-image ResNet/K=3 feature t-SNE control is available in [`strict_domain_resnet_k3_tsne/`](strict_domain_resnet_k3_tsne/).

The strict asynchronous `new_lamp_merge` dataset/backbone grid is available in
[`new_lamp_merge_async_grid/`](new_lamp_merge_async_grid/).

The BloodMNIST non-Fisher baseline experiment is available in
[`blood_nonfisher_baselines/`](blood_nonfisher_baselines/).

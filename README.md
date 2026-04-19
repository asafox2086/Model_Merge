# MedMNSITMerge

这个仓库不负责训练上游模型，它负责消费已经整理好的 `model_hub`，完成三件事：

1. 读取多个 client checkpoint。
2. 执行模型融合。
3. 在 MedMNIST `test` 集上评估并汇总结果。

一句话概括：

- 上游训练并产出 `model_hub`
- 这个仓库做 merge、eval、summary

## 仓库是干什么的

主要入口是：

- [merge.py](/data/liyapeng_grp/program/MedMNISTMerge/merge.py)
  - 单个配置的融合入口
- [evaluate.py](/data/liyapeng_grp/program/MedMNISTMerge/evaluate.py)
  - 单个 merged checkpoint 的评估入口
- [scripts/run_all_avg_eval.py](/data/liyapeng_grp/program/MedMNISTMerge/scripts/run_all_avg_eval.py)
  - 批量执行 `merge + eval`

核心目录是：

- `model_hub/`
  - 上游整理好的输入资产
- `reference_cache/`
  - 为依赖 reference model 的 merge 方法固化的本地参考快照
- `methods/`
  - 所有融合方法实现
- `evaluators/`
  - `small` / `vlm` 的评测逻辑
- `dataset/`
  - MedMNIST `.npz` 数据读取
- `model/`
  - small backbone 和 CLIP 结构重建
- `outputs/`
  - merge 和 eval 产物
- `logs/`
  - 所有批量实验日志

## 我们实现了哪些方法

当前仓库里已经实现的方法有：

- `avg`
- `ties`
- `dare_linear`
- `dare_ties`
- `regmean`
- `fisher`
- `breadcrumbs`
- `model_stock`
- `adamerging`
- `from`
- `iso_c`
- `iso_cts`
- `free_merge`
- `robustmerge`

当前正式对比脚本默认跑下面 12 个方法：

- `avg`
- `ties`
- `dare_linear`
- `dare_ties`
- `regmean`
- `fisher`
- `breadcrumbs`
- `model_stock`
- `from`
- `iso_c`
- `free_merge`
- `robustmerge`

## model_hub 里有什么

`model_hub` 里存的是上游训练好的 client 模型，以及对应实验元信息。

small 目录结构：

```text
model_hub/small/<dataset>/<model>/clients_<n>/beta_<beta>/seed_<seed>/
```

vlm 目录结构：

```text
model_hub/vlm/<dataset>/<clip_model>/clients_<n>/beta_<beta>/seed_<seed>/
```

每个实验目录至少包含：

- `meta.json`
- `client_*.pt`

当前正式实验用到的子集是：

- 数据集：`bloodmnist_224`、`dermamnist_224`、`organcmnist_224`、`organsmnist_224`、`chaoshengmnist_224`
- small backbone：`resnet`、`convnext`、`vit_t`、`swin_tiny`
- vlm：`openai/clip-vit-base-patch32`
- 组合：`clients=3/5/7`，`beta=0/0.01/0.1`

## 结果复现现在是怎么保证的

为了让你在独立对话里直接运行脚本也尽量复现 `result/` 里的数字，仓库现在默认做了两件事：

1. 把依赖 reference model 的方法所需 backbone / CLIP reference 固化到本地 `reference_cache/`
2. 让正式对比脚本和自定义方法脚本共享同一份默认配置来源

为什么要这样做：

- `avg` 这类方法只依赖 `model_hub` 里的 client checkpoint
- `ties`、`dare_*`、`breadcrumbs`、`model_stock`、`from`、`iso_*`、`free_merge`、`robustmerge` 这类方法还会额外构造一个 reference model
- 如果 reference model 临时从外部缓存 / Hub 读取，或者读取失败后退回随机初始化，结果就可能和 `result/` 漂开

现在的默认行为是：

- 批跑脚本启动时会先检查并准备 `reference_cache/`
- 准备完成后，正式运行阶段默认进入“离线复现模式”
- 也就是默认等价于：

```bash
REPRO_MODE=1
HF_LOCAL_FILES_ONLY=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

如果你明确想关闭这个复现模式，允许运行阶段重新访问外部 Hugging Face 资源：

```bash
REPRO_MODE=0 bash run_compare_multi_gpu.sh
```

仓库里用于提前生成本地 reference 快照的工具是：

- `scripts/cache_reference_models.py`

例如只预热 `small`：

```bash
./.gpuenv/bin/python scripts/cache_reference_models.py --model-hub-root model_hub --task-type small
```

例如只预热 `vlm`：

```bash
./.gpuenv/bin/python scripts/cache_reference_models.py --model-hub-root model_hub --task-type vlm
```

如果你已经有了 `reference_cache/`，后续独立对话里直接运行批跑脚本即可，不需要额外手工设置这些离线环境变量。

## 怎么跑对比方法 .sh

正式对比脚本只有一个：

- [run_compare_multi_gpu.sh](/data/liyapeng_grp/program/MedMNISTMerge/run_compare_multi_gpu.sh)

它的行为是：

- 多 GPU 之间并行
- 每张 GPU 内部串行
- 同时覆盖 `small` 和 `vlm`
- 覆盖上面那 5 个数据集、4 个 small backbone、1 个 CLIP
- 覆盖 12 个正式对比方法
- 与自定义方法脚本共用同一份默认变量和方法超参
- 默认先准备 `reference_cache/`，再以离线复现模式运行
- 自动把日志写到 `logs/<RUN_TAG>/formal_compare/`

直接运行：

```bash
cd /data/liyapeng_grp/program/MedMNISTMerge
bash run_compare_multi_gpu.sh
```

默认行为：

- 优先准备本地 `reference_cache/`
- 正式运行阶段只读取本地缓存和本地 reference 资产
- 同一套默认配置会同时作用于正式方法和自定义方法
- 不再依赖“当前对话里是否恰好先跑过一次 reference model”

如果你明确想关闭默认复现模式，允许运行阶段访问外部资源：

```bash
REPRO_MODE=0 bash run_compare_multi_gpu.sh
```

常用环境变量：

- 指定 GPU：

```bash
GPU_IDS="0 1 2" bash run_compare_multi_gpu.sh
```

- 指定输出标签：

```bash
RUN_TAG=formal_20260317 bash run_compare_multi_gpu.sh
```

- 关闭离线复现模式：

```bash
REPRO_MODE=0 bash run_compare_multi_gpu.sh
```

- 保留 `merged.pt`：

```bash
DELETE_FLAG=--no-delete-merged bash run_compare_multi_gpu.sh
```

日志位置示例：

```text
logs/<RUN_TAG>/formal_compare/small__ties__gpu1.log
logs/<RUN_TAG>/formal_compare/vlm__fisher__gpu0.log
```

输出位置示例：

```text
outputs/formal_compare_<RUN_TAG>/small/ties/
outputs/formal_compare_<RUN_TAG>/vlm/fisher/
```

## 怎么设计自己的方法，然后跑自己的 .sh

你如果要加一个自己的方法，例如 `my_merge`，通常改这几处：

1. 在 `methods/` 下新增实现文件，比如 `methods/my_merge.py`
2. 在 [methods/__init__.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/__init__.py) 里注册 import 和别名
3. 在 [merge.py](/data/liyapeng_grp/program/MedMNISTMerge/merge.py) 的 `merge_with_method(...)` 里接入 dispatch
4. 如果有新超参：
   - 改 [merge.py](/data/liyapeng_grp/program/MedMNISTMerge/merge.py) 里的 `METHOD_DEFAULTS`
   - 改 [scripts/run_all_avg_eval.py](/data/liyapeng_grp/program/MedMNISTMerge/scripts/run_all_avg_eval.py) 里的 `parse_args()`

开发自己的方法时，用这个脚本：

- [run_custom_methods_multi_gpu.sh](/data/liyapeng_grp/program/MedMNISTMerge/run_custom_methods_multi_gpu.sh)

它同样是：

- 多 GPU 之间并行
- 每张 GPU 内部串行
- 默认覆盖同一套数据集和模型子集
- 默认和正式对比脚本共用同一份变量、方法超参和复现模式
- 若任务包含 `small` / `vlm`，会先准备对应的 `reference_cache/`
- 日志写到 `logs/<RUN_TAG>/custom_methods/`

最常用的运行方式：

```bash
cd /data/liyapeng_grp/program/MedMNISTMerge
CUSTOM_METHODS="my_merge" bash run_custom_methods_multi_gpu.sh
```

脚本默认方法是 `my_merge`。如果你之前习惯写 `my_method`，现在它也会自动映射到 `my_merge`。

它和正式对比脚本一样，默认会先准备 reference cache，然后以离线复现模式运行。如果你明确想关闭复现模式：

```bash
REPRO_MODE=0 CUSTOM_METHODS="my_merge" bash run_custom_methods_multi_gpu.sh
```

如果你要一次测试多个自己的方法：

```bash
CUSTOM_METHODS="my_merge my_merge_v2" bash run_custom_methods_multi_gpu.sh
```

如果你的方法只支持 `small`：

```bash
CUSTOM_METHODS="my_merge" TASK_TYPES="small" bash run_custom_methods_multi_gpu.sh
```

如果你的方法有额外超参，需要直接透传给批量入口：

```bash
CUSTOM_METHODS="my_merge" CUSTOM_EXTRA_ARGS="--my-alpha 0.3 --my-temp 2.0" bash run_custom_methods_multi_gpu.sh
```

日志位置示例：

```text
logs/<RUN_TAG>/custom_methods/small__my_merge__gpu0.log
logs/<RUN_TAG>/custom_methods/vlm__my_merge__gpu1.log
```

输出位置示例：

```text
outputs/custom_methods_<RUN_TAG>/small/my_merge/
outputs/custom_methods_<RUN_TAG>/vlm/my_merge/
```

## 单个实验怎么跑

单个 merge：

```bash
cd /data/liyapeng_grp/program/MedMNISTMerge
./.gpuenv/bin/python merge.py \
  --config configs/merge/small/blood_resnet_c3_b0_s42.json
```

单个 evaluate：

```bash
cd /data/liyapeng_grp/program/MedMNISTMerge
./.gpuenv/bin/python evaluate.py \
  --config configs/eval/small/blood_resnet_c3_b0_s42.json \
  --merged-dir outputs/merged/small/bloodmnist_224/resnet/clients_3/beta_0/seed_42/avg
```

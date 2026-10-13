# 硬件校准协议 / Hardware calibration protocol

跟踪：[issue #12](https://github.com/FlyTOmeLight/llm-cal/issues/12)。本协议定义测量与验收方法，当前仓库没有据此完成的真实硬件基线。

This protocol specifies measurement and acceptance. No real hardware baseline has been collected with it yet.

## 固定实验条件 / Fix the experiment

每份结果必须记录下表，保持同一轮对比中的模型、请求和系数不变。

Record every field below. Keep model, workload and estimator coefficients identical within a comparison.

| 项目 / Field | 内容 / Required data |
| --- | --- |
| 模型 / Model | Source、model ID、commit SHA、config、checkpoint/quantization；不能只记会变化的分支名。 / Pin the checkpoint and configuration to a commit. |
| 环境 / Environment | llm-cal commit、engine/version、完整启动参数、driver/runtime、GPU ID/数量、互联与节点拓扑。 / Include the estimator commit, engine version, launch flags and hardware topology. |
| 负载 / Workload | 固定 prompt/token IDs、输入与输出 token 数、context 长度、随机种子、并发、prefix caching 设置。 / Fix tokens, lengths, seed, concurrency and prefix-cache settings. |
| 估算 / Estimate | `EvaluationReport` 的参数、权重、KV、prefill/decode 与并发结果；同时保存全部系数。 / Save estimates and all coefficients. |
| 测量 / Measurement | 原始逐请求时间与 token 计数、引擎 KV allocation/usage、GPU memory、失败与重试数；记录采集工具及指标定义。 / Save raw timings, tokens, memory, failures and instrumentation definitions. |

## 测量步骤 / Measurement steps

1. 在目标硬件上启动固定版本引擎。确认模型确实加载成功，再进行测试；保存完整启动日志。 / Start the pinned engine and preserve its model-loading log.
2. 预热后，在并发 1 下测输入 2K、8K、32K token 的请求；每个配置记录至少 30 个完成请求及全部失败。区分冷启动、队列时间、网络时间与引擎时间。 / Warm up, then measure at least 30 completed requests per supported input length, recording every failure and timing boundary.
3. 单请求 decode 记录首 token 之后的输出 token 数与耗时；计算每请求 tokens/s。并发吞吐另外计算总输出 token / 测量窗口秒数，不能混作单请求速率。 / Measure per-request decode separately from aggregate tokens/s across a measurement window.
4. 使用相同模型与负载比较有效 TP=1、2、4、8；显存不足的配置记录无法运行，不用缺失值代替测量。 / Compare valid TP sizes, recording out-of-memory configurations as unavailable measurements.
5. KV 测量使用引擎的已分配 block/token 指标并记录 block 大小、dtype、TP rank 和缓存策略；区分预分配池与实际使用量。显存总使用量的差值不能直接当作 KV 真值。 / Record KV block allocation and actual occupancy per rank, separately from total GPU memory.
6. 覆盖标准 GQA、MLA、NSA/压缩 attention、滑动窗口和 MoE；同一架构至少保留一个独立模型或负载用于验证拟合结果。 / Cover the relevant architectures and hold out an independent model or workload.
7. 逐步增加并发，同时记录每请求速率分布、总体吞吐、TTFT、失败和 OOM。根据事先规定的 SLA 判断可接受并发。 / Increase concurrency while recording rate distributions, TTFT and failures against a predeclared SLA.

## 验收 / Acceptance

- 保存逐配置的 predicted、measured、signed error、absolute relative error；测量为零或不可用时标注原因，不计算误差。 / Retain signed and absolute relative errors, with explicit unavailable measurements.
- 用一组负载拟合利用率或互联系数，再用保留负载验证。拟合和验证结果分开报告。 / Fit on one workload set and validate on held-out workloads.
- 事先声明使用场景与误差阈值。达到阈值后才能声称该范围已校准；未达到时修公式或收窄适用范围。 / Declare the intended workload and acceptable error before the experiment.
- 引擎条目升级为 `verified` 时附测试者、日期、硬件和实际指标，并链接原始结果。一次测试不能证明所有版本、硬件和负载。 / Attach tester, date, hardware, measured metrics and raw evidence before upgrading an engine entry.

建议结果放在 `docs/benchmark-results/`，同时提交可重放的请求数据和采集方式；当前不创建空结果或示例实测数据。

Store actual results under `docs/benchmark-results/` with reproducible workloads and collection instructions. Do not substitute synthetic data for measurements.

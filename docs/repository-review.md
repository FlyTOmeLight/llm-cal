# 仓库评估

评估日期：2026-10-13。范围包括 Python 核心、CLI、Gradio Web、离线测试、三份 YAML 数据和构建/部署工作流。以下内容记录初次评估时的状态。后续已按用户要求修复，当前处理情况见 [issue 进度](issues.md)。

## 总体判断

当前仓库已具备完整的元数据到报告流程。最有价值的基础是实际权重字节、traits 组合、共用 `Evaluator` 和结果来源标签。主要不足集中在跨模块的一致性：计算、解释、有效性判断和多个展示入口没有充分共用证据，模块测试通过仍可能给用户错误的部署建议。

建议先修复数值和输入契约，再补跨模块回归；暂时不以增加模型、GPU 或抽象层为首要目标。

## 实现结构

```text
CLI / Gradio Web / 模型页面生成脚本
  → Evaluator.evaluate()
    → ModelSource.fetch() → ModelArtifact（config、文件大小、commit）
    → 架构 traits → ArchitectureProfile → 参数量与 KV 估算
    → config 或单个 safetensors header → 量化 fingerprint
    → 文件字节汇总与量化 reconciliation
    → GPU 数据 + 引擎兼容矩阵
    → fleet planner → 性能估算 → 并发上界 → 启动命令
  → EvaluationReport → CLI / HTML / Markdown / explain
                        → 可选 LLM 审计
```

模型源通过 ABC 隔离，HF 使用 SDK 获取元数据，ModelScope 使用 httpx REST。领域结果主要是 frozen dataclass，YAML 数据使用 Pydantic。实际执行引擎、下载完整权重和硬件测量均不属于当前评估流程。

## 优先修复的问题

以下 P1 表示直接影响计算结果、部署可行性或凭证边界；P2 表示影响验证可信度、性能或维护。复现均使用 fake source、HTTP patch 和临时缓存，没有请求真实模型或发送审计数据。

### P1：TP decode 吞吐出现二次 GPU 数放大

位置：`src/llm_cal/performance/compute.py:174–197`。

`weight_per_gpu = total_weight_bytes / N` 已使 `effective_bw / weight_per_gpu` 随 N 增大，随后 `cluster_tps = per_gpu_tps × N × efficiency` 再乘一次 N。固定 H100、14 GB 权重和默认系数，1 GPU 得到 107.68 tok/s，4 GPU 得到 1722.86 tok/s，比例为 16。

当前估算接口没有 batch 参数或权重复用模型，这个 N² 放大不能作为 TP 组单请求吞吐，也缺少解释其为批量吞吐的计算依据。并发 L 上界使用该数值，因此也会被放大。

建议明确单请求 tok/s 与批量总 tok/s 的契约，先纠正 TP 组带宽和权重的计算单位。验收：固定系数、忽略额外通信损耗时，单请求的带宽模型随 N 线性增长；并发上界引用与目标 SLA 一致的量。

### P1：非法 TP 仍可显示 fits=True 并生成命令

位置：`fleet/planner.py:103–120,208–269`、`core/evaluator.py:157–181`、`cli.py:44–135`。

用仓库的 Mistral fixture（32 个 attention heads）、H100、14 GB 权重强制 `gpu_count=3`，结果 `fits=True`，理由同时写明 3 不整除头数，命令仍包含 `--tensor-parallel-size 3`。`fits` 只表达容量，而用户需要知道整体配置是否可运行；编排层没有据此阻止不可行命令。

CLI 参数也缺少数值范围和引擎枚举约束。`_generate_command()` 对任何非 sglang 名称都会生成 vLLM 命令。

建议在核心入口校验卡数、token 数、上下文、利用率、SLA 和引擎，并区分容量充足与部署有效。验收：非法 TP 和非法引擎返回明确错误；无法部署的组合不输出可直接执行的推荐命令。

### P1：部分权重大小缺失仍标为 verified 总量

位置：`weight_analyzer/__init__.py:55–79`、`core/evaluator.py:114–125`。

文件大小缺失时通过 `size or 0` 计入总量。样例两分片，大小分别为 100 bytes 和 None，报告返回 `100 [verified]`。用户看不到总量只覆盖已知分片，后续显存规划可能少算权重。对仓库内所有 `.safetensors` 求和也没有区分子目录中的多个权重版本或 adapter，需明确支持的仓库布局。

建议携带元数据完整性，缺失分片大小时输出不完整观测或 unknown，不把部分和当作完整权重。验收：缺失 size 的 fixture 不产生完整 verified 总量或肯定的容量推荐；多套权重的选择遵循明确文件集合。

### P1：Web 凭证隔离依赖全局环境

位置：`web/app.py:959–970,1025–1038,1085–1112,2349–2375`。

`calculate()` 暂时写入 `os.environ` 中的 HF、ModelScope 和 reviewer 凭证，再在 finally 恢复。环境变量属于整个进程；finally 只能恢复值，无法隔离同时运行的请求。代码还启动独立预热线程，该线程同样通过共享 evaluator 和环境读取凭证，即使用户事件串行也仍存在并行读取路径。

这是代码可见的凭证串用风险，本次未做真实多用户泄露实验。共享 evaluator 初始化时也可能将当时环境中的 HF token 保留在客户端中。

建议通过请求级参数将凭证传给 source/header reader/reviewer，后台预热使用独立的服务端配置。验收：两个交错请求及后台预热只使用各自凭证，完成后进程环境不变。

### P1：explain 与实际计算不一致

位置：`core/explain.py:91–125,135–208`、`architecture/formulas/kv_cache.py:68–101`。

Mistral fixture 在 32K context、4K sliding window 下实际 KV 为 536,870,912 bytes；解释先算出 baseline 4,294,967,296 bytes，再直接写 `result = baseline = 536,870,912 bytes`，没有说明窗口截断。NSA 的 top-k modifier 同样没有出现在 explain 路径。

量化解释固定写 argmin 选最优候选，但实现已可由 fingerprint 选择其他方案；解释还指向不存在的 `#params-estimate` 条目。

建议计算结果同时携带关键步骤/证据，渲染层读取该信息；先覆盖实际不一致分支，无需重做全部报告框架。验收：sliding window、NSA、fingerprint 覆盖 argmin 的解释能按步骤得到最终值，标签与来源一致。

## 验证与维护问题

### P2：benchmark 可能“未执行成功却返回成功”

位置：`benchmark/runner.py:80–100,229–232`。

评估异常转换为 SKIP，退出码只看 FAIL。用始终抛异常的 evaluator 跑现有数据集，33 项全部 SKIP，退出码为 0。README 所述“全过 exit 0”并不能由当前退出码保证。

建议区分上游暂时不可用、评估实现出错和预期允许的跳过；全部跳过不能作为 benchmark 成功。验收同时断言通过数量和失败/跳过策略。

### P2：缓存没有减少完整元数据获取

位置：`core/evaluator.py:277–288`、`core/cache.py:49–66`、`web/app.py:2349–2375`。

`_fetch()` 先执行完整 `source.fetch()`，再用取得的 commit 查缓存。固定 commit 连续评估两次，source.fetch 被调用两次；fingerprint header 也未缓存。Web 每种 GPU 比较都会重新评估同一个模型，预热说明声称避免元数据等待，但当前调用顺序无法实现该效果。

commit freshness 的约束本身合理。建议分开轻量 revision 检查与按 revision 获取元数据，并复用同次比较的模型材料。验收以实际 HTTP/source 调用次数为准，仍保持 commit 变化和 refresh 正确失效。

### P2：endpoint 和 revision 没有贯穿请求链

位置：`model_source/huggingface.py:37–41,91–95`、`weight_analyzer/safetensors_reader.py:93–120`、`core/evaluator.py:319–324`。

HF 构造函数保存 `_endpoint`，但 config URL 固定到 huggingface.co；header reader 默认也使用该域名，evaluator 没有传 endpoint。HF SDK 的元数据路径与后续 HTTP 路径可能不同。ModelScope 在无 commit 时获取 config/tree 使用 `master`，header 请求却由 evaluator 统一退到 `main`。

建议源适配器统一管理有效 endpoint/revision，并让 header 获取复用该配置。验收检查每次请求 URL/revision，无需真实联网。

### P2：测试集中在模块，关键用户链路缺少覆盖

222 项离线测试全部通过，但本次 branch coverage 为 60%。关键模块覆盖率：evaluator 33%、explain 20%、HF source 20%、formatter 12%、concurrency 34%。Web 不在 coverage source 和 CI 的 lint/type 检查范围内。没有 `test_evaluator.py`、`test_huggingface_source.py` 或 Web 行为测试；HF 模块注释提到的 `tests/test_hf.py` 实际不存在。

建议优先增加 fake source 驱动的端到端报告测试，围绕上述缺陷和错误路径组织，而非以覆盖率数字为目标。现有 decode 测试主要比较 NVLink 型号差异，未约束 TP 缩放；benchmark 测试覆盖字段比较和退出策略，却未实际调用 run_all。

### P2：现有门禁和发布链路未闭合

本次执行 `ruff check src tests` 报 11 项问题，集中在 `tests/test_modelscope_source.py`；`ruff format --check src tests` 有 5 个文件不符合格式。mypy 通过。CI 安装未锁定工具版本，因此本次结果代表当前允许安装的版本，不等同于历史版本的 CI 状态。

Web 部署和 PyPI 发布工作流独立运行，没有依赖 CI 成功的 job 门禁；源码推送可以在测试结果未确认时触发 Web 发布。`web/requirements.txt` 同时安装 llm-cal 和 vendored source，存在重复安装；`.github` 的 Web 部署 path filter 未包含根 `pyproject.toml`。

建议先修复已有门禁失败，再让发布依赖同一提交的验证结果，统一部署依赖来源。验收：失败提交不能发布，部署所需依赖变更会触发构建。

### P2：文档、依赖和证据约束有漂移

- ADR-001 已决定不引入 ModelScope SDK，但 `pyproject.toml` 仍把 `modelscope>=1.20.0` 放在生产依赖；当前源实现没有使用它。按现有决策删除无用依赖可直接减少安装内容。
- README 仍将 per-tensor dtype 列为后续能力，而 fingerprint/header reader 已实现；贡献指南仍写 ModelScope 决策 pending，且要求 CLI 小于 60 行，当前为 207 行。
- `EngineCompatEntry` 未跨字段验证 `verified` 必须有 tested 证据、`cited` 必须有 URL/date；`EngineSource` 没有 metrics 字段。当前矩阵没有 verified 条目，但 schema 不能阻止之后的错误升级。
- 来源标签对单分片 dtype 推导的量化方案一律升级为 verified，未区分直接声明与抽样推断。应在评估范围和标签中保留这一区别。

建议先清理现状与已接受 ADR 的冲突，再用少量 schema 校验保护已明确的证据规则；模型/GPU/引擎数量直接从数据统计，避免在多个入口维护硬编码现状。

## 数学模型仍需独立校准

上面的代码缺陷可在本地复现；架构公式与引擎实际内存布局的一致性还需要独立证据。当前 benchmark 主要比较权重、架构和部署参考值，不能证明实际 KV、延迟或并发精度。

值得优先校准的是 MoE prefill（当前按所有参数计算 FLOPs）、MLA/NSA 的 KV 存储、混合全局/滑动注意力，以及非 NVIDIA 互联的性能系数。`profile` 参数在 prefill/decode 中未用于相应架构建模，单一 NVLink 指标也不足以描述所有厂商的互联。

后续按模型 revision、引擎版本、硬件、负载和测量方法记录证据，再决定需要增加哪些公式分支。未完成硬件测量前，保持 estimated 和适用范围，不把手算一致当作运行时验证。

## 本次实际验证

- 临时虚拟环境：CPython 3.11.17，按 `.[dev]` 安装仓库声明的依赖。
- `pytest -q --cov=llm_cal --cov-report=term-missing`：222 passed，branch coverage 60%。
- `mypy src`：通过，44 个 source files。
- Ruff 0.17.0：lint 11 项问题，format 5 个文件待格式化。
- 离线行为样例：确认 TP N² 缩放、非法 TP 命令、缺失权重 size 的 verified 标签、sliding-window explain 矛盾、全 SKIP benchmark 成功退出、缓存前完整 fetch。
- 未运行真实模型 benchmark、Gradio 浏览器交互、硬件推理、打包发布或部署。

推荐实施顺序：修复数值/有效性与凭证隔离 → 补相应链路回归和 explain → 修缓存/endpoint/benchmark → 清理门禁、依赖与文档 → 校准真实硬件模型。

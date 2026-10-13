# llm-cal 项目协作规则

本文件适用于整个仓库，补充当前生效的全局规则。项目事实以当前实现、测试、数据文件和已接受的 ADR 为准；文档有冲突时先核对这些证据。仓库不足与改进建议见 [仓库评估](docs/repository-review.md)，该评估不是已授权的实施计划。

## 项目目标与执行入口

llm-cal 根据模型元数据估算推理所需的硬件、KV cache、性能和并发，并生成引擎启动命令。模型文件大小是观测值，部署容量和性能是估算值；保持两者的来源和可信度清楚可见。

- Python 包采用 `src/` 布局，入口 `llm-cal = llm_cal.cli:app`。版本由 Git 标签和 `hatch-vcs` 推导。
- `core/evaluator.py` 的 `Evaluator.evaluate()` 是 CLI、Gradio Web 和模型文档生成脚本共用的编排入口，返回 `EvaluationReport`。
- 默认评估会访问模型源；它读取 config、文件元数据，并在需要时请求一个 safetensors 分片的 header。执行普通评估、benchmark 或文档生成前，先判断任务是否需要网络。
- `web/app.py` 是单独部署的 Gradio 入口；从仓库根目录运行 `LLM_CAL_PREWARM=0 python web/app.py` 可关闭启动时的后台联网预热。安装 Web 依赖见 `web/requirements.txt`。

## 按改动定位模块

| 改动对象 | 实现与相关约束 |
| --- | --- |
| 模型获取与鉴权 | `model_source/`：`ModelSource.fetch()` 返回 `ModelArtifact`，源适配器负责网络和错误映射，架构解释留给下游。HF 文件大小通过 `model_info(files_metadata=True)` 一次取得。ModelScope 使用 REST + httpx，改动前读 [ADR-001](docs/adr/001-modelscope-integration-strategy.md)。 |
| 架构与公式 | `architecture/`：`detector.py` 组合 `traits.py` 的独立特征，生成 `ArchitectureProfile`；`formulas/` 计算 KV 和参数量。新增架构前读 [架构指南](docs/architecture-guide.md)。 |
| 权重与量化 | `weight_analyzer/`：汇总文件大小，`fingerprint.py` 识别 config/header 证据，`reconciler.py` 对比量化方案。区分显式声明、抽样识别和 bytes/param 猜测。 |
| 硬件与引擎数据 | `hardware/gpu_database.yaml`、`engine_compat/matrix.yaml`；对应 `loader.py` 用 Pydantic 校验。新增 GPU 或引擎支持优先改数据。 |
| GPU 数与并发容量 | `fleet/planner.py`：单节点 TP 候选、权重分配、KV 分片和三档推荐；`performance/concurrency.py`：容量与吞吐上界。保持每 GPU 与整个 TP 组的量纲一致。 |
| 性能 | `performance/compute.py`：prefill、decode、利用率和互联系数。改公式或系数前读 [方法论](docs/methodology.md)，用独立推导或测量验证。 |
| 输出与说明 | `output/formatter.py` 输出 CLI 报告，`core/explain.py` 输出推导，`common/i18n.py` 管理 CLI 翻译；Web 和模型页面有各自渲染函数。 |
| 启动命令 | `command_generator/`：vLLM、SGLang 命令。核对 TP、上下文长度、引擎兼容条目和 flag，命令字符串不得作为代码执行。 |
| LLM 审计 | `llm_review/reviewer.py`：用户显式开启后发送推导链；结果是独立意见，不改变主报告数值和标签。 |
| 验证与文档生成 | `tests/` 为离线测试；`benchmark/dataset.yaml` 为参考数据；`scripts/generate_model_pages.py` 联网生成中英文模型页。 |

## 项目契约

### 来源与标签

- 新增面向用户的数值使用 `output.labels.AnnotatedValue` 携带标签和来源；各渲染入口保持相同含义。
- `verified` 表示直接读到的数据，`inferred` 表示推导或猜测，`estimated` 表示公式估算；引用使用 `cited`，缺少验证使用 `unverified`，无法判断使用 `unknown`。
- 缺失字段、未知架构、缺失文件大小和证据冲突要在结果中保留不确定性。抽样 header 的证据范围是该分片，不能自动代表整个模型。
- `[llm-opinion]` 是 opt-in 审计输出的独立标签。审计失败时仍保留主报告。
- 引擎条目标为 `verified` 必须附真实硬件测试证据和指标；`cited` 必须附 URL 和获取日期。维护 YAML 时检查证据内容，不能只依赖当前 schema。

### 计算与展示

- 架构能力通过 traits 组合表达。修改识别优先级、压缩比例或滑动窗口规则时，检查组合场景和 `tests/test_detector.py`、`tests/test_formulas.py` 的关键回归。
- GB 与带宽 GB/s 的数值转换遵循实现的十进制 `1e9`；保留 bytes、tokens、seconds 等中间单位，明确单卡、TP 组、单请求和并发吞吐。
- 公式修正时同步核对 `core/explain.py`、CLI formatter、Web 渲染和模型页生成器；解释中的输入、步骤、最终值和标签必须对应同一次计算。
- 现有单节点规划上限和经验系数是当前实现边界，修改它们需要具体需求和验证证据。不要把公式结果描述为实测性能。
- CLI/Web 保留输入与展示职责，计算与有效性判断集中在核心模块。Web 用户凭证应按请求传入依赖，不能依赖修改进程环境来隔离用户。
- 用户可见的文字和数据说明同时检查英文与中文。新增标签需同步图例、翻译和全部渲染入口。

## 验证与开发

开发环境与门禁定义见 `pyproject.toml`、[贡献指南](CONTRIBUTING.md) 和 `.github/workflows/ci.yml`。基本安装与检查：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]' -r web/requirements.txt
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
```

- CI 覆盖 Python 3.11、3.12、3.13；代码提交需满足上述 lint、格式、严格类型和测试门禁。文档小改动按实际范围检查链接、命令和事实。
- 测试中的 config 使用 `tests/fixtures/configs/`；网络边界用 fake source 或 HTTP stub 隔离，临时缓存用 pytest 的 `tmp_path`。`responses` 面向 requests，现有 httpx 测试使用 `patch("httpx.get", ...)`。
- 修复跨模块问题时验证 `Evaluator → EvaluationReport → 输出/命令` 的可观察行为；数学回归的预期值用独立手算，避免复制实现后互相证明。
- 标有 `CRITICAL:` 的回归覆盖核心契约。保留它们的保护范围；改变预期前先证明旧契约为何错误。
- `llm-cal --benchmark` 是联网参考检查，不等于 `pytest`；同时查看 PASS/FAIL/SKIP 数量，不能仅根据退出码声明通过。
- 数据文件通过 `importlib.resources` 加载并随包发布。改动打包配置或新增运行时资源时，验证构建出的 wheel 能读取这些资源。
- CI 对 `src`、`tests` 执行 lint/type/test，并编译 Web/脚本；Web 行为测试在 pytest 中执行，交互改动仍需按范围验证。部署工作流会复制 `web/` 入口和 `src/llm_cal`，同一提交验证通过后，推送 `main` 会触发对应发布；PyPI 版本发布入口见 `publish.yml`。

## 文档维护

- 新增模型能力、CLI 参数、公式和来源约定时，同步相关中英文说明。已有说明可能滞后，先核对实现，不把路线图当成现状。
- `docs/models/` 与 `docs/zh/models/` 的模型页由脚本生成；改版式优先修改生成器。重新生成会联网并改写多页，只在任务需要时执行。
- 将长期协作约束留在本文件，将具体缺陷、复现证据和改进优先级留在评估或 issue 中，避免把缺陷写成必须保持的行为。

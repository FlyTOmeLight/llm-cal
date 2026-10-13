# Issue 处理进度

以下代码修复已在本地完成，尚未合并或部署。GitHub issues 保持开放，合并修复后再关闭。

| Issue | 本地处理 |
| --- | --- |
| [P1：TP decode 吞吐出现二次 GPU 数放大](https://github.com/FlyTOmeLight/llm-cal/issues/1) | 已修复，包含离线回归或构建验证 |
| [P1：非法 TP 仍可显示 fits=True 并生成命令](https://github.com/FlyTOmeLight/llm-cal/issues/2) | 已修复，包含离线回归或构建验证 |
| [P1：部分权重大小缺失仍标为 verified 总量](https://github.com/FlyTOmeLight/llm-cal/issues/3) | 已修复，包含离线回归或构建验证 |
| [P1：Web 凭证隔离依赖全局环境](https://github.com/FlyTOmeLight/llm-cal/issues/4) | 已修复，包含离线回归或构建验证 |
| [P1：explain 与实际计算不一致](https://github.com/FlyTOmeLight/llm-cal/issues/5) | 已修复，包含离线回归或构建验证 |
| [P2：benchmark 可能“未执行成功却返回成功”](https://github.com/FlyTOmeLight/llm-cal/issues/6) | 已修复，包含离线回归或构建验证 |
| [P2：缓存没有减少完整元数据获取](https://github.com/FlyTOmeLight/llm-cal/issues/7) | 已修复，包含离线回归或构建验证 |
| [P2：endpoint 和 revision 没有贯穿请求链](https://github.com/FlyTOmeLight/llm-cal/issues/8) | 已修复，包含离线回归或构建验证 |
| [P2：测试集中在模块，关键用户链路缺少覆盖](https://github.com/FlyTOmeLight/llm-cal/issues/9) | 已修复，包含离线回归或构建验证 |
| [P2：现有门禁和发布链路未闭合](https://github.com/FlyTOmeLight/llm-cal/issues/10) | 已修复，包含离线回归或构建验证 |
| [P2：文档、依赖和证据约束有漂移](https://github.com/FlyTOmeLight/llm-cal/issues/11) | 已修复，包含离线回归或构建验证 |
| [P2：建立性能与 KV 公式的真实硬件校准基线](https://github.com/FlyTOmeLight/llm-cal/issues/12) | 已提供测量协议；真实硬件校准待完成 |

校准协议见 [calibration.md](calibration.md)。

本次验证：271 项 pytest 通过；Ruff lint/format、mypy、actionlint、sdist/wheel 构建和 MkDocs strict 构建通过。打包从 sdist 构建 wheel，使用临时环境中的构建后端（`--no-isolation`）。

未运行真实模型 benchmark 或 GPU 推理；预生成模型页仍为原有快照，下一次文档部署会使用修复后的生成器重新计算，任何模型评估失败都会阻止发布。

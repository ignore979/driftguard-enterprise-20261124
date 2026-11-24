# DriftGuard 企业赛 Harness

2026 年 11 月 24 日锁定版，使用 Contest Harness SDK 1.0.2 和 `contest.harness.workspace.v2`。C 发动机任务从平台选中的父检查点进行工况残差续训，按公开验证和 PyTRiO 宿主网关回执选择候选。A 气体和 B 轴承使用自研校赛版 SDK 训练逻辑。A/B 实验候选独立于评测镜像。

## 镜像对应关系

已交付的归档为 `driftguard-enterprise-linux-amd64-20261124.tar`。镜像内实际入口 `/app/contest_submission.py` 来自 [企业赛入口](submission/enterprise_contest_submission.py)。[发布校验清单](RELEASE_MANIFEST.json) 记录归档及参与评测的 16 份源码的 SHA-256，包括官方 SDK 启动模板。

C 的十轮本地 SDK 演练使用官方固定公开验证和真实 PyTRiO Sampler，第 1、2、5、6、10 轮采用候选。最终 RMSE 为 50.8212 循环，MAE 为 36.2137 循环。企业赛线上成绩以评测平台回执为准。这个镜像的归档、层、应用文件与宿主 SDK `describe` 已检查，Linux 容器导入和启动待验证。

## 复现

- Docker 镜像依赖按 [镜像清单](requirements-image.txt) 固定。通过比赛渠道取得 SDK 1.0.2，通过正常依赖渠道取得 Linux/amd64、CPython 3.12 兼容的 wheels。[SDK 来源](SDK_SOURCE.json) 和发布清单记录版本、下载出处及校验值。
- 使用 `python submission/stage_enterprise_context.py --wheel-dir <wheels目录> --output <新的构建目录>` 核验并生成上下文。以清单中的官方 Python Linux/amd64 OCI 基础镜像作为 `python_base` 构建上下文，运行 `docker buildx build --platform linux/amd64 --load --network none --build-context python_base=oci-layout://<基础镜像布局> -t driftguard-enterprise:20261124-locked <构建目录>`。重新构建的镜像归档摘要取决于构建器和归档元数据。
- 宿主将公开投影和父检查点挂载到 SDK 约定位置。默认入口为 `python3 /app/harness.py describe`。评测时宿主调用同一入口的 `train` 和 `predict`，并绑定 PyTRiO Sampler。
- 本地实验用官方数据包。依据 [输入摘要](docs/official-inputs.json) 核对原始文件，将目标任务的三个输入文件及 profile 放在数据工程的 `data/<任务>/`，把输入摘要复制为该工程的 `data/manifest.json`。C 公开固定对照可执行 `python scripts/run_c_rounds.py --source-root <数据工程> --strategy condition`，分类配对可以执行 `python scripts/run_a_study.py --data-root <数据工程>`。B 历史对照和 PyTRiO 续训还需对应的团队历史检查点和授权账号。
- Python 源码位于 `src/`，公开数据对照、回放、PyTRiO 训练及 SwanLab 记录工具位于 `scripts/`。[来源声明](NOTICE.md) 说明代码与数据归属。

仓库发布源代码、配置和校验信息。赛事数据、模型权重、语料、账户密钥和镜像归档通过各自授权渠道提供。

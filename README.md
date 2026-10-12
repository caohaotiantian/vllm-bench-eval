# vllm-bench-eval

用 **`vllm bench serve`** 压测任意 OpenAI 兼容推理服务，并把**测评用例**和**测评结果**同步到内网的 **Benchmark 平台**。

关注指标：**TTFT**、**TPOT**、**有效吞吐 TPS（输出 TPS / 总 TPS）**、**E2E 总时间**。

整个工具是可移植的：把这个目录拷到任意服务器，**只需要改 `config.yaml` 一个文件**，然后跑一条命令。

---

## 1. 安装

```bash
cd vllm-bench-eval           # 本仓库目录（包名是 vllm-bench-eval）
uv sync                      # 创建 .venv 并安装依赖
uv run vllm-bench-eval --help
```

`pyproject.toml` 里用 `[[tool.uv.index]]` 把默认索引钉在 `https://pypi.org/simple`，
这样 `uv.lock` 不会被某台机器的私有镜像地址污染、换台服务器仍能装。
要用自己的镜像：`UV_DEFAULT_INDEX=<url> uv sync`。

依赖里 **不包含 vllm**（macOS 没有 wheel，而且被测服务器通常已自带）。
真正跑压测有两种方式，见「使用场景」。

### 关于平台 SDK 版本

> Benchmark 平台的 Python SDK 在 PyPI 上以 `opik` 这个包名发布，所以
> `pyproject.toml` 里的依赖写的是 `opik>=1.9.8,<2`——那是**包名**，不是平台名称。

**SDK 大版本必须与 Benchmark 平台服务端匹配**——用 2.x 的 SDK 去打 1.x 的平台服务端，
会在 dataset 相关接口上报 `HTTP 404`。

如果你有平台 SDK 的本地源码，可以改用本地 SDK：

```bash
# 方式 A：临时装进当前 venv（注意之后再跑 `uv sync` 会被覆盖）
uv pip install -e /path/to/platform-sdk/sdks/python

# 方式 B：写进 pyproject.toml（持久，但会降低可移植性）
# [tool.uv.sources]
# opik = { path = "/path/to/platform-sdk/sdks/python", editable = true }
```

`pyproject.toml` 末尾已经留好了方式 B 的注释模板。

---

## 2. 配置

```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml
```

`config.yaml` 是**唯一**需要改的文件。五个小节：

| 小节 | 作用 | 关键字段 |
| --- | --- | --- |
| `server` | 被测的 OpenAI 兼容服务 | `base_url`（不带 `/v1`）、`endpoint`、`backend`、`model`、`tokenizer` |
| `benchmark` | `vllm bench serve` 参数 | `dataset_name`、`dataset_path`、`num_prompts`、`request_rate`、`max_concurrency`、`percentile_metrics` |
| `runner` | 怎么执行压测 | `mode: native \| docker`、`capture`、`python`、`docker_image`、`docker_platform` |
| `benchmark_platform` | Benchmark 平台目标 | `url`（平台 API 地址，结尾要有 `/api/`）、`workspace`、`project_name`、`dataset_name`、`experiment_name` |
| `prepare` | `prepare-dataset` 的默认值 | `source`、`num_samples`、`output_path` |

两个容易踩的点：

* **`server.model` 必须与服务端 `/v1/models` 返回的 id 完全一致。**
* **`server.tokenizer` 必填**：`vllm bench serve` 需要一个 HuggingFace tokenizer 来统计
  输入/输出 token 数（TPOT、TPS 都依赖它）。填一个与被测模型同族的公开 tokenizer 即可，
  例如 `Qwen/Qwen3-8B`。**离线/内网环境**可以直接填本地目录（含 `tokenizer.json` /
  `tokenizer_config.json` 的模型目录），例如 `/models/Qwen3-8B`，这样不会去连 HF Hub。
* **`server.base_url` 填服务根地址、不要带 `/v1`**（`/v1` 属于 `endpoint`）。写了会被自动去掉。
* **`benchmark_platform.url` 是 Benchmark 平台的 API 地址，结尾必须是 `/api/`**。
  请填写实际部署的 API 地址，例如 `https://benchmark.example.com/api/`。

任意字段都能用环境变量覆盖，便于 CI：

```bash
VBE_SERVER__MODEL=qwen3-8b VBE_BENCHMARK__NUM_PROMPTS=32 uv run vllm-bench-eval run
```

`benchmark_platform` 这一节对应的前缀是 `VBE_BENCHMARK_PLATFORM__*`。

**底层 SDK 环境变量兼容**：`OPIK_URL_OVERRIDE` / `OPIK_WORKSPACE` / `OPIK_API_KEY` /
`OPIK_PROJECT_NAME` / `HF_TOKEN` 也会作为对应字段的回退被读取（`VBE_*` 优先级更高）。
本工具**不读也不写 SDK 的用户级配置文件 `~/.opik.config`**，平台地址只来自配置文件/环境变量。

---

## 3. 命令

```bash
uv run vllm-bench-eval check                  # 检查推理服务 / 数据集 / runner / 平台连通性
uv run vllm-bench-eval prepare-dataset        # 下载官方测评集切片，写成 JSONL，并上传到平台 Dataset
uv run vllm-bench-eval run                    # 压测 + 同步（主命令）
uv run vllm-bench-eval sync results/xxx.json  # 只同步一个已有的结果文件
uv run vllm-bench-eval show results/xxx.json  # 只解析并打印指标，不连平台
```

常用开关：`run --no-sync`（只压测）、`run --allow-failures`（有失败也退出 0）、
`run --experiment-name NAME`、`prepare-dataset --num-samples 32 --no-upload`。

`check` 是**只读**的：它不会在平台上创建 Dataset，只汇报是否存在。

---

## 4. 三种使用场景

### 场景 A：目标服务器上已经装了 vLLM（native，**默认**）

这是 `config.example.yaml` 的默认配置。

```yaml
runner:
  mode: native
  capture: true
  python: null        # 必须能 import vllm；null = 当前解释器
```

> **重要**：`capture: true`（默认）时本工具用
> `python -m vllm_bench_eval.vllm_entry` 启动压测，所以**那个 python 必须能
> `import vllm`**。两种做法二选一：
> 1. 把 vLLM 装进本工具的 venv：`uv pip install vllm`（Linux 上有 wheel）；
> 2. 或者让 `runner.python` 指向已装 vLLM 的那个环境的解释器，例如
>    `python: /opt/vllm-venv/bin/python`（该入口只依赖标准库 + vllm，不需要本工具的其它依赖）。
>
> 如果两者都做不到，设 `capture: false` 退回普通 `vllm bench serve`——能出指标，
> 但 trace 没有时间轴、没有 span（见下文「采集」）。

```bash
uv run vllm-bench-eval run
```

### 场景 B：本机没有 vLLM（docker）

vLLM 在 macOS 上没有 wheel，但官方发布了 `manylinux2014_aarch64` / `manylinux1_x86_64` wheel，
所以可以用容器只跑**客户端**压测逻辑（不需要 GPU、不需要模型权重）。

```bash
docker build -t vllm-bench-client:0.11.0 -f docker/Dockerfile docker/
```

把 `config.yaml` 的 `runner` 换成（`config.example.yaml` 里已有注释掉的模板）：

```yaml
runner:
  mode: docker
  docker_image: vllm-bench-client:0.11.0
  docker_platform: null            # Apple Silicon 上设为 linux/arm64
  docker_user: auto                # Linux 上以当前 uid:gid 运行
```

```bash
uv run vllm-bench-eval run
```

容器会自动：

* 用 `--add-host host.docker.internal:host-gateway` 访问宿主机，
  `server.base_url` 里的 `localhost`/`127.0.0.1` 会被自动改写成 `host.docker.internal`；
* 挂载 `data/`（只读）、`results/`（写结果）、`.cache/huggingface`（缓存 tokenizer）；
* 通过 `--name` 命名，`runner.timeout_sec` 超时会连容器一起 kill；
* `docker_user: auto` 在 **Linux** 上加 `--user $(id -u):$(id -g)`，避免 `results/`、
  `.cache/` 被写成 root 所有（macOS 上 Docker Desktop / OrbStack 会自动映射，故为 no-op）。

> **Linux 上的两个前提**：
> 1. 被测服务必须监听 `0.0.0.0`（而不是只监听 `127.0.0.1`），否则容器无法通过
>    `host.docker.internal` 连上；
> 2. `--add-host=host.docker.internal:host-gateway` 需要较新的 Docker（20.10+）。
> 也可以改用 `docker_extra_args: ["--network", "host"]` 并把 `host_gateway_alias`
> 设成 `127.0.0.1`。

`server.api_key` / `HF_TOKEN` 只以**变量名**传进容器（`-e OPENAI_API_KEY`），
值走进程环境，不会出现在命令行里，回显的命令也会做脱敏。

镜像里用的是 `/usr/local/bin/vllm-bench-serve`（见 `docker/vllm_bench_serve.py`）而不是
`vllm bench serve`：`vllm` 主 CLI 会为**所有**子命令构建参数解析器，其中 `vllm serve`
会实例化 `VllmConfig` 并要求一个可用的加速器，在纯 CPU 容器里会直接报
`Failed to infer device type`。该封装只调用 `vllm.benchmarks.serve` 的 `add_cli_args` +
`main`，**参数和行为与 `vllm bench serve` 完全一致**。

### 采集：为什么 trace 需要 `capture`

`vllm bench serve` 的结果 JSON 里**没有** prompt、没有请求 ID、也没有任何墙钟时间戳。
只靠它同步出来的 trace 会全部落在"同步那一刻"、`end_time` 为空、没有 span——
时间轴完全没有可用性。

所以 `capture: true` 时，本工具用自己的入口
`vllm_bench_eval/vllm_entry.py` 启动压测：它包装
`vllm.benchmarks.lib.endpoint_request_func.ASYNC_REQUEST_FUNCS` 里的每个请求函数，
把每次调用的输入（prompt、prompt_len、期望输出长度、model、api_url、采样参数）和
输出（success、latency、ttft、itl 列表、生成文本、output_tokens、error）连同
**墙钟起止时间**追加到 `<结果文件>.requests.jsonl`，然后照常调用
`vllm.benchmarks.serve.main()`——**压测参数与行为和 `vllm bench serve` 完全一致**。

细节：

* vLLM 在正式压测前会先发一个**预热请求**。它构造 `RequestFuncInput` 时**不带
  `request_id`**，而每个被计量的请求都带——因此 `request_id is None` 就是一个
  精确、且不依赖版本的判别条件。预热记录会被打上 `is_warmup` 标记并在解析时丢弃，
  同步时还会核对条数是否等于 `num_prompts`。
* `RequestFuncOutput.start_time` 是 `perf_counter()`，不是墙钟，所以入口自己记
  `time.time()`。
* docker 模式把本包**只读挂载**进容器（`/opt/vbe`）跑同一个入口，
  采集逻辑只有一份，不会与镜像里的副本漂移。

### 场景 C：只同步已有的结果

已经在别的机器上跑过 `vllm bench serve`（必须带 `--save-result --save-detailed`
且 `--percentile-metrics` 含 `e2el`），把 JSON 拷过来：

```bash
uv run vllm-bench-eval sync /path/to/result.json
```

如果结果文件旁边有对应的 `<结果文件>.requests.jsonl`（即当初是 `capture: true` 跑的），
把它一起拷过来，就能得到完整时间轴和 span；否则只会同步聚合指标。

没有 sidecar 时，要让每个请求关联回 Dataset item，本地的 `benchmark.dataset_path` /
`benchmark.dataset_name` 必须与产出该结果时一致（**`benchmark.seed` 不影响对齐**，
原因见下文「逐请求对齐」）。

---

## 5. Benchmark 平台中的数据模型

| 平台对象 | 内容 | 对应指标 |
| --- | --- | --- |
| **Dataset**（`benchmark_platform.dataset_name`） | 每条 benchmark prompt 一个 item：`sample_id`、`line_index`、`prompt`、`dataset_source` | —— 这就是「benchmark 用例同步到平台」 |
| **Experiment**（`benchmark_platform.experiment_name`） | 一次压测一个，挂在上面的 Dataset 上。`experiment_config` = `{run_id, metrics: 全部聚合指标, run: 运行配置, capture, runner_mode}` | 聚合 TTFT/TPOT/TPS/E2E 全量留档 |
| **Trace `req-0000`…** | 每个请求一条，**带真实起止时间** | 见下 |
| **Span `预填充(TTFT)` / `解码`** | 每条 request trace 下两个 span | 把 TTFT 阶段与解码阶段在时间轴上分开 |
| **Experiment item** | 把每条 request trace 与它用的 Dataset item 关联 | 在 Experiment 页面可逐用例对比 |
| **Trace `压测汇总`** | 每次压测一条汇总 trace，时间跨度 = 整轮压测窗口 | 聚合指标 |

### 命名约定：展示名中文，行业术语保留英文

平台上**展示**给人看的名字（feedback score 名、汇总 output 的键、span 名、汇总 trace 名）
一律中文，但 **TTFT / TPOT / ITL / P50·P90·P99 / tokens/s / req/s / ms / s**
这些行业通用写法保留英文——翻译反而更难读。

**机器字段保持英文**，方便脚本、diff 和看板稳定：原始结果 JSON（vLLM 自己的格式）、
`experiment_config`（`metrics` / `run`）、所有 trace 与 span 的 `metadata` 键、采集 sidecar。

所有展示名只在 `vllm_bench_eval/metric_names.py` 一处定义。

### 每个请求的 Trace

* 名字 `req-0000`…（序号，不是文案），`start_time` / `end_time` 是**真实墙钟**，
  `duration` ≈ 该请求的 E2E 延迟。并发效果直接能在时间轴上看出来
  （`max_concurrency: 2` 时前两条重叠，第三条在其后开始）。
* `input` = `{prompt, max_tokens, model}`
* `output` = `{generated_text, success, output_tokens}`
* `tags` = `[平台 tags…, model_id, backend, dataset_name]`
* `metadata`（**英文键**）= `run_id`、`experiment_name`、`model_id`、`tokenizer_id`、
  `backend`、`vllm_version`、`endpoint`、`base_url`、`request_rate`、`max_concurrency`、
  `request_index`、`request_id`、`sample_id`、`input_tokens`、`output_tokens`、
  `max_output_tokens`、`ttft_ms`、`tpot_ms`、`e2e_ms`、`sampling_params`、`ignore_eos`、
  `itl_stats`（`{count, mean, p50, p90, p99, max}`）、`itl_ms`（完整列表，**上限 512 条**，
  超出时 `itl_truncated: true`）、`error`
* `feedback_scores`：只留 5 个。token 计数移到 metadata / span usage，不污染分数统计。

| 平台上显示的名字 | 内部 key | 含义 |
| --- | --- | --- |
| `TTFT(ms)` | `ttft_ms` | 首 token 时间 |
| `TPOT(ms)` | `tpot_ms` | 每输出 token 时间（不含首 token） |
| `端到端延迟(ms)` | `e2e_ms` | 该请求端到端耗时 |
| `输出吞吐(tokens/s)` | `output_tokens_per_s` | 该请求的输出吞吐 |
| `请求成功` | `success` | 1 = 请求成功完成 |

### 两个 Span

| Span | 时间范围 | 携带 |
| --- | --- | --- |
| `预填充(TTFT)` | 请求发出 → 第一个 token | `type=llm`、`ttft_ms`、`input_tokens`、`model`、`provider` |
| `解码` | 第一个 token → 最后一个 token | `type=llm`、`tpot_ms`、`output_tokens`、`itl_stats`、`decode_ms`、`model`、`provider`、**`usage = {prompt_tokens, completion_tokens, total_tokens}`** |

`usage` 让平台的 token / 成本列自动填充，并会汇总到 trace 级别。

### 汇总 Trace `压测汇总`

* `start_time` / `end_time` = 第一个请求开始 → 最后一个请求结束。
* `input` = 运行配置（model / tokenizer / backend / endpoint / base_url /
  dataset / num_prompts / request_rate / max_concurrency / vllm_version / tool_version）。
* `output` = **分组**后的指标，而不是一堆平铺的 key：

  ```json
  {
    "TTFT(ms)":      {"均值": …, "中位数": …, "p50": …, "p90": …, "p99": …},
    "TPOT(ms)":      {…},
    "ITL(ms)":       {…},
    "端到端延迟(ms)": {…},
    "吞吐": {"请求(req/s)": …, "输出(tokens/s)": …, "总计(tokens/s)": …},
    "计数": {"完成请求数": …, "请求总数": …, "失败请求数": …,
             "输入token总数": …, "输出token总数": …, "总耗时(s)": …}
  }
  ```

* `metadata` = 完整原始结果（**英文键**，去掉逐请求大数组）。
* `feedback_scores`：13 个头部指标。

| 平台上显示的名字 | 内部 key |
| --- | --- |
| `TTFT均值(ms)` / `TTFT P50(ms)` / `TTFT P99(ms)` | `mean/p50/p99_ttft_ms` |
| `TPOT均值(ms)` / `TPOT P50(ms)` / `TPOT P99(ms)` | `mean/p50/p99_tpot_ms` |
| `端到端延迟均值(ms)` / `端到端延迟 P50(ms)` / `端到端延迟 P99(ms)` | `mean/p50/p99_e2el_ms` |
| `输出吞吐(tokens/s)` | `output_throughput_tps` |
| `总吞吐(tokens/s)` | `total_token_throughput_tps` |
| `请求吞吐(req/s)` | `request_throughput_rps` |
| `请求完成率` | `completed_ratio` |

> **按分数名筛选**：平台的查询语言用点号取键，名字里带括号时要**加引号**：
> `feedback_scores."TTFT(ms)" > 100`、`feedback_scores."端到端延迟(ms)" > 1000`；
> 不带括号的可以不加引号：`feedback_scores.请求成功 = 1`。

> 术语说明：vLLM 的 `e2el` 是**单个请求**的端到端延迟；`duration` 才是整轮压测的
> **E2E 总时间**。两者都同步了。

## 6. 逐请求对齐（trace 怎么关联到正确的用例）

**有 sidecar 时（默认）**：直接用采集到的 **prompt 原文**做精确匹配。
不需要复算 vLLM 的洗牌，也不依赖它的实现细节，并且对**任意**数据集类型都成立——
`sharegpt` / `random` / `hf` 跑出来的 prompt 同样会被采集到，工具会把它们
（按内容去重）补充进 Dataset，于是这些数据集也能得到关联好的 experiment item。

**没有 sidecar 时（`capture: false`）**：退回到复算 vLLM 洗牌的老办法，且只对
`custom` 数据集有效。`vllm bench serve --save-detailed` 写出的数组是按**请求发出顺序**
排列的，而 `CustomDataset` 会：

```python
# vllm/benchmarks/datasets.py
random.seed(self.random_seed)      # get_samples() 构造时没传 random_seed，
random.shuffle(self.data)          # 因此恒为 DEFAULT_SEED = 0
```

`random.shuffle` 的置换只取决于列表长度和种子，所以
`samples.custom_dataset_order(n, k, seed)` 能精确复算（含 `num_prompts >` 样本数时的
`random.choices` 补采样）。

> **注意：这条回退路径恒用 `DEFAULT_SEED = 0`，与 `benchmark.seed` 无关。**
> `--seed 0` 和 `--seed 42` 得到的顺序都是 `[4, 1, 5, 2, 0, 3, 7, 6]`。
> 因此 `align_requests_to_samples()` **不接受 seed 参数**——传了 `benchmark.seed`
> 会算出另一个置换却依旧报告"对齐成功"，等于给每条 trace 贴错标签。

`sample_id` 由「行号 + prompt 内容的 sha1 前 12 位」组成，改了内容就会得到新的 id，
不会和平台上旧的 item 撞号。

## 7. vLLM 版本兼容

`vllm bench serve` 的参数是逐版本加上去的，写死一套就会在老版本上直接报错：

```
vllm bench serve: error: unrecognized arguments: --custom-skip-chat-template
```

所以本工具**每次运行前先探测一次**目标 vLLM 支持哪些参数，把它不认识的**自动去掉**，
并为每个被去掉的参数打一行 WARN 说明「少了它会怎样」。探测结果在进程内缓存，
一次运行只探一次。

* 探测方式：`capture: true` 时跑 `python -m vllm_bench_eval.vllm_entry --vbe-probe`
  （一次调用同时拿到**版本号**和完整参数表）；`capture: false` 时跑
  `vllm bench serve --help` 并解析其中的 `--flag`。docker 模式在容器内做同样的事。
* **探测失败不会挡住压测**：会退回"全部照发"的老行为，并打一行 WARN。
* 探测结果只有在**包含全部必需参数**时才采信 —— 否则（例如 `docker run` 失败时打出的
  是它自己的 usage）就当作探测失败，避免把真参数全删光。
* **重命名过的参数会自动换拼写**：`--backend` 在 0.9.2 之前叫 `--endpoint-type`，
  工具检测到后会改用旧名并 WARN，而不是当作"不支持"丢掉。
* 必需参数（`--base-url --endpoint --model --dataset-name --dataset-path
  --num-prompts --save-result --result-dir --result-filename`）如果缺失，
  会直接报 `RunnerError` 并带上探测到的版本号。
* 探测到的版本会写进 `experiment_config.vllm_version`、每条 trace 的
  `metadata.vllm_version` 和汇总 trace。

### 各参数的引入版本（对着 PyPI sdist 逐版本核对）

| 能力 | 引入版本 | 缺失时的影响 |
| --- | --- | --- |
| `--dataset-path` | **0.9.1** | 0.9.0 无法指定数据集文件 → 直接报错 |
| `--save-detailed` | **0.9.1** | 结果 JSON 没有逐请求数组；逐请求指标只能来自 capture sidecar |
| **`--dataset-name custom`** | **0.9.2** | 更早的版本压根不支持"用自己的 prompt"→ 直接报错 |
| `--backend` | **0.9.2** | 更早的版本叫 `--endpoint-type`，工具会**自动改用**旧拼写 |
| `--custom-output-len` | **0.9.2** | 用数据集默认的输出长度 |
| `--custom-skip-chat-template` | **0.9.2** | tokenizer 的 chat template 会套到每个 prompt 上，实际发出去的文本与数据集 prompt 不一致 |
| `--ready-check-timeout-sec` | **0.10.1** | 就绪检查超时用 vLLM 默认值 |
| `RequestFuncInput.request_id` | **0.10.2** | 见下方"采集的降级" |

**最低可用版本 `0.9.2`** —— 这是第一个支持 `--dataset-name custom` 的版本，
而"把自己的 prompt 作为测评集"正是本工具的核心用法。0.9.0 / 0.9.1 会得到一条
明确的 `RunnerError`（列出该版本支持哪些 dataset 名、并提示升级），而不是
argparse 的 `invalid choice` 崩溃。
**推荐 `0.10.2+`**：采集的预热识别从启发式变为精确。

> 参数**值**也会被检查，不只是参数名：`--dataset-name` 的可选值是逐版本变的
> （0.9.0 只有 `random`，0.9.1 有 `sharegpt/burstgpt/sonnet/random/hf`，
> 0.9.2 起才有 `custom`），探测时会一并读出来。

### 采集（capture）在老版本上的降级

* **模块路径**：`endpoint_request_func` 在 0.10.1 从 `vllm/benchmarks/` 挪到了
  `vllm/benchmarks/lib/`。采集入口两条路径都会尝试。
* **预热请求识别**：0.10.2+ 靠"预热请求没有 `request_id`"来精确识别；更老的版本
  根本没有这个字段，于是退回"**第一次调用就是预热**"（预热在主循环开始前被 await，
  所以必然是第 0 次调用）。同步时会用 `num_prompts` 复核：如果按这个猜测剔除后条数不对、
  而全部保留反而正好等于 `num_prompts`（例如就绪检查被跳过、压根没有预热），
  就撤销这次猜测。
* **`RequestFuncOutput.start_time`** 直到 0.11.0 才有，而且是 `perf_counter()` 不是墙钟——
  采集入口一直用自己记的 `time.time()`，所以与版本无关。

### 指标缺失是被容忍的

老版本没有 `--percentile-metrics ... e2el` 时结果里不会有 e2el 聚合。解析器、
分组输出和汇总 trace 都会**跳过缺失的指标块**而不是报错；极端情况下（完全没有分位数指标）
汇总 trace 的 output 只剩 `throughput` 和 `counts`。

## 8. 已知限制

* **老版本 vLLM 会少若干能力**（见上一节的表），工具会自动降级并逐条 WARN，不会崩。
* **`capture: false` 时 trace 没有时间轴、没有 span**，并且只有 `dataset_name: custom`
  能对齐（靠复算 vLLM 的洗牌）。`sharegpt` / `random` 等在这种模式下不会与 Dataset item
  关联（`run` 会打印 WARN）。开启 `capture`（默认）即可解决这两点。
* **`capture: true` 要求跑得起 `import vllm` 的 python**（见「场景 A」）。
* **逐请求 E2E**：有 sidecar 时直接用 vLLM 自己测的 `latency`；没有时按 `ttft + sum(itl)`
  推导（在 vLLM 源码里是恒等式，但只在**流式**后端成立）。
* **`itl_ms` 在 metadata 里最多存 512 条**，超出时 `itl_truncated: true`；
  完整分布始终可从 `itl_stats` 读到。
* **流式响应常常不带 `usage`**，此时逐请求 `output_tokens` 由 vLLM 重新分词得出
  （取自结果 JSON 的 `output_lens`），可能与服务端自己的计数略有出入。
* **逐请求 TPOT 需要 `output_tokens > 1`**，否则该 score 不会上报。
* **Docker 模式的时延包含容器网络开销**（`host.docker.internal` 多一跳 NAT）。做绝对值对比时
  建议在目标服务器上用 native 模式。
* **平台 SDK 与平台服务端大版本必须匹配**（见上文）。
* `experiment_name` 重复不会报错，平台允许同名 Experiment 共存；建议模板里保留 `{ts}`。
* 本工具**不**启动被测服务，也不管理模型；`server.base_url` 指向的服务需要提前起好。
* **失败请求不上报延迟指标**：vLLM 对失败请求记 `ttft=0`、`itl=[]`，直接同步会把平台上的
  平均值拉向 0。所以失败请求只上报 `success=0` 和 token 计数，不上报
  `ttft_ms`/`tpot_ms`/`e2e_ms`/`output_tokens_per_s`。
* **只要有请求失败或 `completed < num_prompts`，`run` 就会打印 WARN 并以非 0 退出**
  （结果照样同步）。要在有失败时仍然退出 0，加 `--allow-failures`。
* **非 `custom` 数据集 + 平台上尚不存在该 Dataset** 时 `sync` 会直接报错，而不是建一个
  空 Dataset + 没有 item 的 Experiment。请改用 `dataset_name: custom`，或把
  `benchmark_platform.dataset_name` 指向已有 Dataset，或 `--no-sync`。
* **平台 Dataset 里出现重复 `sample_id` 会报错**（关联关系会变得不确定）：换一个
  `benchmark_platform.dataset_name`，或清掉旧 item。

## Coding agent 多轮任务测评

`agent-run` 执行真实的多轮闭环：模型读取代码、修改文件、执行命令，然后根据工具结果继续请求。
原有 `run` 和 `sync` 仍然用于 vLLM 单请求性能测评。

```bash
# 先将示例配置的 server.model 改为服务端模型名称。
uv run vllm-bench-eval agent-run -c examples/coding-agent/config.example.yaml --no-sync
```

模型服务必须支持 OpenAI Chat Completions 的 `messages`、`tools` 和 `tool_calls`。
每次响应必须提供 `usage.total_tokens`。缺少该字段时任务失败，以避免绕过 token 预算。
`stream: true` 开启流式请求并测量 TTFT、TPOT 与请求总耗时。
默认 `stream: false` 保留非流式兼容模式，其 TTFT/TPOT 为空。

任务 JSONL 的每一行格式如下：

```json
{"task_id":"repair-add","prompt":"Repair add.py so the tests pass.","repository":"repository","verify_command":["python3","verify_add.py"],"verification_files":["verify_add.py"]}
```

`task_id` 在同一数据集中必须唯一。`repository` 相对任务 JSONL 文件目录解析。
其他配置路径相对 YAML 文件目录解析。每个任务获得新的仓库副本。任务内部按依赖顺序执行，任务之间由 `max_concurrency` 配置调度。
源仓库不接受符号链接，副本排除 `.git`、`.venv` 和 `__pycache__`。
工作目录必须位于所有源仓库外部。其余文件会复制，因此应使用专用测评仓库。

模型工具包括 `read_file`、`write_file` 和 `run_command`。
文件工具仅访问任务副本内的路径，并拒绝绝对路径与符号链接越界。
`run_command` 接收 `argv` 数组，不通过 shell 拼接命令。
程序内置 `allow_any_command: false` 时，除精确 `verify_command` 外，额外模型命令必须匹配 `command_allowlist` 参数前缀。
例如 `[python3, verify_add.py]` 允许该脚本及其后续参数。
不要把 `[python3]` 或 `[sh]` 视为安全限制：这些前缀允许执行任意程序。

`allow_commands` 默认开启。模型可执行任务 `verify_command` 的精确 argv，额外参数不会自动获得权限。
受限模式下，额外模型命令需匹配 `command_allowlist`。设置 `allow_commands: false` 会禁止此实时测评流程的命令与最终验证。
验证命令由数据集作者确定，模型不能替换它。
`verification_files` 指定不得修改的验证文件；数据集作者应列出验证脚本、测试与配置依赖。
程序还自动保护 `verify_command` 参数直接引用的现有仓库文件。
每次工具调用后及最终验证前后，程序比较原始文件内容哈希。
验证文件被修改、删除或替换为越界链接时，任务以 `verification_tampered` 失败。
固定的内联验证命令无需文件，但其导入的验证依赖仍需显式列入保护列表。
该检测防止正常工具执行篡改评分文件，不能防御拥有宿主权限的程序攻击测评进程。
命令具有宿主用户的权限，并继承宿主环境。仓库副本不是安全沙箱。
对不可信模型、代码或任务，需在外部受限容器或虚拟机内运行整个测评程序。
命令超时时，程序终止其进程组。输出截断会保留 `truncated` 标记。

任务只有在模型正常结束、验证命令退出码为零、预算未超限且工具覆盖达标时通过。
默认要求在整个任务会话中实际调用 `read_file`、`write_file` 和 `run_command`，不要求在同一轮调用。
成功的文件工具调用和实际启动的允许命令满足覆盖要求；非零命令退出码单独计入工具失败。
拒绝执行的命令不计入覆盖。`required_tools: []` 可关闭覆盖要求，任务中的同名字段优先于配置。
模型声称成功、HTTP 请求成功、工具成功或代码出现修改都不单独构成任务通过。
验证命令决定代码评分质量；任务通过还必须满足实际工具覆盖要求。
若任务无需修改代码，可显式设置 `required_tools: []` 关闭覆盖要求，再由验证结果判定通过。
默认遇到任务失败后 CLI 退出码为 1。使用 `--allow-failures` 可保留结果并返回 0。

| 配置字段 | 作用 |
|---|---|
| `max_turns` | 每个任务的模型请求数量上限 |
| `max_tool_calls` | 每个任务的工具调用数量上限 |
| `max_tokens_per_turn` | 每次请求的最大输出 token 数 |
| `token_budget` | 所有轮次的累计 `usage.total_tokens`，含重复输入上下文 |
| `request_timeout_sec` | 单次模型请求超时 |
| `task_timeout_sec` | 模型与工具执行的任务总超时，不含初始仓库复制 |
| `command_timeout_sec` | 工具命令与最终验证命令超时 |
| `max_output_chars` | 文件读取与命令输出的字符上限 |

结果保存在 `agent-<run_id>.json`，事件保存在同名 `.events.jsonl`。
事件增量写入并刷新至磁盘，包含任务、轮次、模型调用、工具调用和验证的开始与结束。
进程强制终止后，已写事件仍然存在，未结束步骤可能只有开始事件。
v2 结果同时保存每任务检查点，可按下述安全边界恢复。事件含完整对话和工具内容，应按测评数据管理。

开启 `benchmark_platform.enabled` 后，`agent-run` 会同步数据集、实验和任务级 trace。
建议使用独立的 `dataset_name`。每个任务根 trace 包含轮次、LLM、工具和验证子 span。
任务成功率百分比与任务总耗时作为反馈评分，实验项关联完整任务根 trace。
可以稍后同步已保存结果：

```bash
uv run vllm-bench-eval agent-sync results/agent-<run_id>.json -c examples/coding-agent/config.example.yaml
```

同步需要原始事件文件仍然位于结果内 `events_path` 指定的位置。
任务数据项按内容去重，trace 与 span 使用稳定 ID。
重复同步仍然创建新的实验记录，当前不保证实验级幂等。


### 长会话配置、指标与恢复

示例配置启用流式输出、最多 100 轮和 2 个任务并发。
同一任务的轮次与工具按依赖顺序执行，不裁剪消息历史。
`max_concurrency` 限制任务并发，不改变每个任务内部顺序。
文件读取和命令输出仍受 `max_output_chars` 限制，截断内容带显式标记。

流式指标采用客户端测量，定义如下：

| 指标 | 定义 |
|---|---|
| TTFT | 发出请求到首个非空内容或工具函数名、参数增量的时间，包含网络与服务端等待 |
| TPOT | `(请求E2E - TTFT) / (completion_tokens - 1)`，与原 vLLM 的主口径一致 |
| `chunk_tpot_ms` | 首尾有效输出分片间隔除以 `completion_tokens - 1`，保留为客户端分片估计 |
| 请求 E2E | 发出请求至完整响应结束，包含最终 usage 和结束帧 |
| 单请求输出吞吐 | `completion_tokens / 请求E2E秒数` |
| 单请求总吞吐 | `total_tokens / 请求E2E秒数` |
| `chunk_itl_ms` | 相邻有效 SSE 输出分片的时间间隔，不是逐 token 精确 ITL |

角色、心跳和仅工具 ID/type 的元数据不会触发 TTFT。
无有效输出时 TTFT 和 TPOT 为空。缺少输出 token 数或其值小于 2 时 TPOT 为空。
结果保留空值原因；分位数不会把空值作为零。
`sample_counts` 保存各指标的有效样本数，便于判断长会话的指标覆盖范围。
主 TPOT 包含最后有效分片之后的协议耗时。`chunk_tpot_ms` 不包含这段耗时，不能与主 TPOT 混用。
指标定义版本为 `metric_definition_version: 2`；读取与同步旧结果时会按当前公式重新计算，不改写源文件。
缺少 DONE、结束原因、完整工具参数或有效 JSON 时，流式请求失败，不执行不完整工具。

Opik 中一个任务对应一个根 trace，其下包含轮次、LLM、工具和验证 span。
模型 token usage 只放在 LLM span，根 trace 与汇总不重复报告 usage。
LLM span 保留原有 TTFT、TPOT、端到端延迟、输出吞吐和请求成功展示名。
根 trace 记录任务成功率百分比、含工具的任务耗时、工具次数、失败次数和 token 计数。
实验项关联的任务根 trace 还包含该任务全部正常 LLM 请求的 TTFT、TPOT、E2E 均值与 P50/P90/P99。
`请求完成率` 表示正常模型请求占比，不受最终代码验证失败影响。
`单请求输出吞吐均值/P50/P90/P99(tokens/s)` 与对应总吞吐统计使用每个模型请求自己的 E2E 分母。
`任务工作负载输出/总吞吐(tokens/s)` 使用包含工具与验证的任务活动窗口，不能视为纯推理服务吞吐。
工具与验证保留各自 span 时间、调用数、失败数及未结束数，不再展示总耗时汇总指标。
`压测汇总` trace 与实验配置包含请求分位数、吞吐、请求数量和完成率。
汇总保留原有 `输出吞吐(tokens/s)`、`总吞吐(tokens/s)` 名称，其 metadata 明确标注 agent 工作负载范围。
工具失败与模型请求失败分别记录。代码任务通过与请求完成不等价。

并发吞吐使用整个执行窗口，不累加各任务耗时。
恢复后的总吞吐使用所有执行尝试的活动窗口之和，排除两次启动间的停机时间。
`attempt_metrics` 另外保存每次执行尝试的指标。
分位数仅纳入正常完成的模型请求；失败请求仍计入请求总数和失败数。

固定轨迹回放用于比较不同模型、服务参数或推理配置的长上下文性能：

```bash
uv run vllm-bench-eval agent-run -c examples/coding-agent/config.example.yaml \
  --replay-result results/agent-<source_run_id>.json --no-sync
```

也可设置 `mode: replay` 与 `replay_result_path`。
回放重发 v2 事件中完整完成的 LLM 请求，保留录制消息、工具结果及请求参数。
模型名称与流式模式采用当前配置，服务地址与 endpoint 采用当前配置。
新输出不会替换后续请求中的录制历史，模型提出的新工具也不会执行。
回放创建新的 run、trace 和性能记录，源代码与源工作副本不会改变。
回放状态为 `replayed`，`success` 为 null，仅评价请求执行，不评价代码任务成功率。
回放同样受任务、轮次、请求及 token 预算约束。
v1 结果仍可读取和同步，但缺少完整请求参数及安全检查点，不能回放或恢复。

恢复一个未完成的 v2 实时任务：

```bash
uv run vllm-bench-eval agent-run -c examples/coding-agent/config.example.yaml \
  --resume-result results/agent-<run_id>.json --no-sync
```

也可设置 `resume_result_path`。恢复与回放不能同时开启。
恢复保留原工作副本、消息历史、已用 token、验证哈希和累计活动耗时。
原任务文件、模型、工具权限和预算必须保持一致；并发数可以调整。
服务地址与凭据可调整；结果不会保存服务凭据。
完成任务直接跳过，不重复执行模型或验证命令。

完整轮次结束后，程序原子写入安全检查点。
模型与工具执行前，程序先写入 running 标记并刷新至磁盘。
若检查点之后仅有模型请求，恢复可在新 attempt 下重试该轮。
若检查点之后存在任何工具或验证开始事件，恢复拒绝执行，即使该操作已经结束。
这是因为工具可能已产生副作用，不能安全地自动重复它。
缺少工作副本、检查点身份不一致或验证文件改变时，恢复也会拒绝。
需要人工检查不确定的副作用，或创建新的独立任务运行。
同一运行使用进程级文件锁。第二个并行恢复进程会被拒绝，避免重复执行工具。

未完成请求的服务端 token 用量可能未知，无法恢复准确的计费用量。
已记录的完整 usage 会计入预算；重试使用新的 span ID，保留旧请求证据。
程序收到取消后停止新任务，已运行命令终止其进程组。
非流式 HTTP 请求可能需要等待请求超时后才能结束取消。
结果、事件和检查点需要一起保留。移动文件后需保持结果中的绝对路径有效。

### 服务参数与平台同步

`coding_agent.extra_body` 用于附加服务请求参数，并保存至录制请求与运行配置。
回放保留原始服务参数；恢复要求这些参数与原运行一致。
该字段不能覆盖消息、工具、模型、流式设置、输出预算等核心控制字段。
凭据必须使用 `server.api_key` 或其环境变量，不要放入 `extra_body`。

Opik 要求平台 trace/span 使用 UUIDv7。同步会按录制时间确定性转换内部 ID，并统一转换父子关系与实验引用。
内部 ID 和原始记录保持不变，平台 metadata 保存来源 ID，因此旧结果可以重新同步。
同步会检查 SDK 的 flush 状态和本次同步期间报告的后台投递错误。
后台批次被拒绝或 flush 超时时，CLI 返回非零；仅队列清空不能证明服务器已接收数据。
修复平台问题后，可使用 `agent-sync` 重新同步已有结果，无需再次调用模型。


### 从 vLLM 支持的编码题完成测评

[官方 vLLM CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) 支持 `hf` 数据集 `openai/openai_humaneval`。
本项目把[固定 OpenAI HumanEval 原始数据](https://github.com/openai/human-eval/tree/463c980b59e818ace59f6f9803cd92c749ceae61)转换为多轮工具任务。
这属于同源题目的 agent 工作负载，不是原生 vLLM 单轮压测，也不代表完整 HumanEval 质量评测。
默认仅使用 HumanEval/0 与 HumanEval/1，用于检查完整链路。

每题生成独立仓库，包含 `solution.py`、原始测试组成的 `verify.py` 与许可说明。
标准答案 `canonical_solution` 不写入模型请求、工作副本或任务数据。
`provenance.json` 与任务 provenance 保存源版本、源 SHA256、题目 ID、样本哈希和抽样规则。
原始验证文件受哈希保护。模型必须读取文件、写入实现并通过允许命令运行测试。
生成目录已存在时程序拒绝覆盖，可通过 `--output-dir` 选择新的目录。

使用完整注释模板 [vllm-humaneval.example.yaml](examples/coding-agent/vllm-humaneval.example.yaml)。
先配置被测服务的地址、模型与 endpoint，并填写测评平台的 API 地址、工作区、项目和数据集。
如果服务需要认证，通过 `VBE_SERVER__API_KEY` 设置凭据。
随后准备两题并运行完整测评：

```bash
uv run vllm-bench-eval prepare-agent-dataset -c examples/coding-agent/vllm-humaneval.example.yaml
uv run vllm-bench-eval agent-run -c examples/coding-agent/vllm-humaneval.example.yaml
```

准备阶段只下载公开数据并生成本地任务，不调用模型，也不写入 Opik。
运行阶段调用被测服务、执行工具和原始测试，并创建平台数据集、实验与完整任务 trace。
在 CLI 查看 `2/2 passed`、结果路径和同步报告，并在 Opik 检查两个实验项。
结果 `tool_types` 应包含三种工具，`effective_required_tools` 显示本次覆盖要求。
若模型没有调用必要工具，即使测试通过，任务也以 `required_tools_missing` 失败。

质量反馈使用 `任务成功率(%)`：单个任务为 0 或 100，汇总与实验配置采用已结束实时任务的平均百分比。
失败请求、验证失败和工具覆盖失败的已结束任务进入分母；pending、中断待恢复和回放任务不进入分母。
回放不发布任务质量成功率。`请求完成率` 仍是 0 到 1 的模型请求指标，不与任务成功率混用。
工具与验证总耗时不再作为反馈评分或汇总输出，逐工具 span 时间仍然保留。

需要其他数量或固定随机抽样时，可使用 `--num-samples` 与 `--seed`。
修改输出目录后，也应修改 `coding_agent.tasks_path` 指向新的 `tasks.jsonl`。
本地 HumanEval JSONL 或 gzip 可通过 `--source-file` 输入，不需要下载或安装 datasets/pyarrow。
更改服务地址、模型与 endpoint 后，可用相同任务比较不同部署的 vLLM。

旧检查点没有工具覆盖审计时，恢复沿用原有要求，不补加新默认覆盖约束。
旧 trace 的历史评分不会自动删除；重新同步会追加或更新当前评分，新运行只发布当前名称。


### 允许任意命令

公开 agent 配置样例将 `allow_commands` 与 `allow_any_command` 都设为 true，复制样例后可运行任意有效 argv。
程序内置默认值保持不变：省略字段时 `allow_commands: true`、`allow_any_command: false`，只允许精确验证命令与配置前缀。
若需要受限模式，应在样例中显式改为 `allow_any_command: false`。
若任务需要运行任意命令，可配置：

```yaml
coding_agent:
  allow_commands: true
  allow_any_command: true
  command_allowlist: []
```

也可通过环境变量临时开启：

```bash
VBE_CODING_AGENT__ALLOW_ANY_COMMAND=true uv run vllm-bench-eval agent-run -c examples/coding-agent/vllm-humaneval.example.yaml
```

`allow_commands: false` 始终禁止命令，任意命令开关不能覆盖它。
模型会收到当前命令权限说明，`run_command` 仍接收 `argv` 数组，不自动拼接 shell。
无效 argv、超时、进程组终止和验证文件保护规则保持有效。
任意命令具有宿主用户权限；对不可信代码，应在受限容器或虚拟机内运行测评。
该权限记录在运行配置中，恢复时不允许改变。旧 v2 记录缺少此字段时按 false 处理。

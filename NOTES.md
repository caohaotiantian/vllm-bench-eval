# NOTES —— 设计决策与假设

## 1. 为什么 docker 模式要自带一个 `vllm-bench-serve` 封装

`vllm` 主 CLI (`vllm/entrypoints/cli/main.py`) 在 `main()` 里为**所有**子命令构建 argparse
解析器，其中 `vllm serve` 的解析器会实例化 `VllmConfig` → `DeviceConfig.__post_init__` →
`current_platform.device_type`。PyPI 的 vLLM wheel 是 CUDA build，在纯 CPU 容器里平台探测失败，
于是 `vllm bench serve --help` 都跑不起来：

```
RuntimeError: Failed to infer device type
```

`docker/vllm_bench_serve.py` 绕开主 CLI，直接调用
`vllm.benchmarks.serve.add_cli_args(parser)` + `main(args)` —— 这正是
`BenchmarkServingSubcommand` 内部做的事，**参数与行为与 `vllm bench serve` 完全一致**。
native 模式仍然调用真正的 `vllm bench serve`。

镜像构建过程中踩到的三个坑（都已固化进 `docker/Dockerfile`）：

1. `libgomp.so.1 not found` —— torch/torchaudio 需要，`apt-get install libgomp1`。
2. `Qwen2Tokenizer has no attribute all_special_tokens_extended` —— vLLM 0.11.0 只声明
   `transformers>=4.55.2`，但与 Transformers v5 不兼容，必须 `transformers<5`。
3. `ImportError: Please install vllm[bench] for bench support` —— `CustomDataset` 用 pandas，
   需要 `vllm[bench]` extra。

`Failed to import from vllm._C with ImportError('libcudart.so.12: ...')` 这条 WARNING 是**预期的**：
客户端压测不需要 CUDA kernel。

## 2. 逐请求 → 数据集用例的对齐是怎么保证的

`--save-detailed` 的数组按**请求发出顺序**排列（`serve.py` 里
`outputs = await asyncio.gather(*tasks)`，tasks 按 `input_requests` 顺序创建，`gather` 保序）。
而 `input_requests` 不是文件行序：`CustomDataset.load_data()` 做了
`random.seed(self.random_seed); random.shuffle(self.data)`。

关键发现：`datasets.py:1348` 的 `get_samples()` 里构造的是
`CustomDataset(dataset_path=args.dataset_path)` —— **没有传 `random_seed`**，
所以 shuffle 恒用 `BenchmarkDataset.DEFAULT_SEED = 0`，与 `--seed` 无关。

`random.shuffle` / `random.choices` 的结果只取决于（列表长度, 种子），因此
`vllm_bench_platform.samples.custom_dataset_order(n, k, seed)` 能精确复算这个置换，包括
`num_prompts > 样本数` 时 `maybe_oversample_requests` 的 `random.choices` 补采样。
`tests/test_samples.py::test_custom_dataset_order_matches_vllm_shuffle` /
`…_replicates_oversampling` 用真实的 `random` 调用对拍。

代价：这依赖 vLLM 的实现细节。如果换成非 `custom` 数据集，`align_requests_to_samples` 会返回
False，工具照样同步聚合指标，但不建立 trace ↔ dataset item 关联，并打印 WARN（不会静默出错）。

## 3. 逐请求 E2E 与 TPOT 的推导

结果 JSON 不保存逐请求 `latency`。在 `vllm/benchmarks/lib/endpoint_request_func.py` 中：

* `output.ttft = first_token_timestamp - st`
* 每个后续 token 追加 `itl.append(timestamp - most_recent_timestamp)`
* `output.latency = most_recent_timestamp - st`

因此 `latency == ttft + sum(itl)` 是恒等式，本工具据此计算 `e2e_ms`，再用
`tpot = (e2e - ttft) / (output_len - 1)` 得到逐请求 TPOT（`output_len <= 1` 时不上报）。
聚合的 `mean/p99_e2el_ms` 等直接取自 vLLM 自己算好的字段，不做二次推导。

## 4. Benchmark 平台相关

> 命名约定见第 9 节：对外一律称 **Benchmark 平台**；下面提到的 SDK 包名/环境变量
> 是依赖标识，不是产品名。

* **不读也不写 SDK 的用户级配置文件**（本机上它是过期的，指向 5173）。地址只来自
  `config.yaml` 的 `benchmark_platform.url`、`VBP_BENCHMARK_PLATFORM__URL`
  或 SDK 环境变量回退，通过 SDK 的 `host=` 参数传入。
* **`source` 是平台 `DatasetItem` 的保留字段**（只接受 MANUAL/TRACE/SPAN/SDK），
  所以样本来源写在 `dataset_source` 里。
* **SDK 大版本必须与平台服务端匹配**：PyPI 上 2.2.66 的 SDK 打这台机器上的平台服务端时，
  dataset 接口返回 `HTTP 404`；降到 `>=1.9.8,<2`（实测解析到 1.11.14）即正常。
  `pyproject.toml` 因此把 SDK 钉在 `<2`，并在注释里留了指向本地 SDK 源码
  (`…/agent_benchmark/sdks/python`) 的 `[tool.uv.sources]` 模板。
* `dataset.insert()` 不返回 item id，且按内容 hash 去重。所以流程是：
  插入 → `get_items()` → 用自定义的 `sample_id` 字段建 `sample_id -> dataset_item_id` 映射。
  重复 `run` 不会产生重复 dataset item。

## 5. 本机可行性验证用的具体配置

| 项 | 值 |
| --- | --- |
| 被测服务 | LM Studio, `http://127.0.0.1:1234`，`/v1/completions`，backend `openai`（流式） |
| 模型 | `qwable-9b-claude-fable-5` |
| tokenizer | `Qwen/Qwen3-8B`（公开的同族 tokenizer，仅用于 token 计数；LM Studio 的模型 id 在 HF 上没有对应仓库） |
| 测评集 | vLLM 官方 `benchmarks/sonnet.txt`，每 8 行合成一个 prompt，确定性抽 8 条 |
| 参数 | `--num-prompts 8 --max-concurrency 2 --custom-output-len 128 --custom-skip-chat-template` |
| runner | docker，`vllm-bench-client:0.11.0`（`linux/arm64`，vLLM 0.11.0） |
| Benchmark 平台 | `http://localhost:5174/api/`，workspace `default`，无 API key |

`--custom-skip-chat-template` 是有意开启的：这样发给 `/v1/completions` 的就是 prompt 原文，
平台 trace 里 input 与实际请求体一致，对齐关系也更容易解释。
如果被测的是 chat 接口，应改成 `backend: openai-chat` + `endpoint: /v1/chat/completions`
并关掉该开关。

## 6. 其他假设

* `results/` 与 `.cache/` 已 gitignore；`data/` 里保留了一份小样本，离线也能跑。
* `config.yaml` 也在 gitignore 里（含站点相关信息），仓库里交付的是 `config.example.yaml`。
* 没有初始化 git 仓库（按要求）。

## 7. 对齐的实证验证（2026-09-17，本机）

在容器里直接调用 vLLM 自己的 `get_samples()`（同样的 JSONL、`num_prompts=8`、
`custom_skip_chat_template=True`），把它实际采样到的 prompt 顺序打出来，与本工具复算的
置换逐条比对：

* 复算得到的文件行序 `[4, 1, 5, 2, 0, 3, 7, 6]`，prompt 顺序与 vLLM 的 `get_samples()`
  **完全一致**；
* 结果 JSON 里的 `input_lens = [92, 91, 93, 89, 81, 85, 88, 78]` 与 vLLM 自己算的
  `prompt_len` **逐项相等**；
* 逐请求推导的 E2E 均值 `6727.757953871 ms` 与 vLLM 自己输出的
  `mean_e2el_ms = 6727.757406748 ms` 相差 5e-7 ms，佐证了 `latency == ttft + sum(itl)`；
* 平台里抽查 experiment item：dataset item 的 prompt 是
  `Full many a glorious morning have I seen…`（十四行诗 33 首），对应 trace 的输出是
  `Even so my sun today (now fair as he)…`，正是该诗的续写——语义上也能对上。

---

## 8. 评审后的修复（changelog）

### BLOCKER

1. **对齐用错了 seed**。`cli._sync` 把 `benchmark.seed` 传给了
   `align_requests_to_samples`，但 vLLM 的 `CustomDataset` 恒用 `DEFAULT_SEED=0`
   （`get_samples()` 构造时不传 `random_seed`）。`seed: 42` 时工具会算出
   `[3,4,6,7,2,5,0,1]` 而真实顺序仍是 `[4,1,5,2,0,3,7,6]`，于是**每条 trace 都贴错标签，
   却照样打印「OK aligned」**。
   修复：`align_requests_to_samples()` **彻底删掉 seed 参数**（不是改默认值——留着参数
   就还能被传错），内部固定用 `VLLM_DEFAULT_DATASET_SEED`。`benchmark.seed` 只用于
   `--seed`。README 里「seed 必须一致」的说法一并改正。
   新增测试：`test_alignment_ignores_benchmark_seed`（用 `inspect.signature` 断言没有该参数，
   并断言两个 seed 的置换确实不同）、`test_alignment_uses_default_seed_even_for_a_seeded_run`。

### SHOULD-FIX

2. **native 模式也要 HF 环境**。新增 `runner.build_env(cfg)`，两种模式共用：
   native 下 `HF_HOME` 指向 `runner.hf_cache_dir` 的绝对路径，docker 下指向 `/hf-cache`；
   `HF_TOKEN` 两种模式都注入。测试：`test_native_mode_also_gets_hf_env`、
   `test_docker_mode_points_hf_home_at_the_mount`。
3. **`OPENAI_API_KEY` 用 `setdefault` 导致外部导出的值压过配置**。改为无条件赋值。
   测试：`test_configured_api_key_beats_a_preexported_one`。
4. **密钥出现在 argv 里并被回显**。docker 改为 `-e OPENAI_API_KEY` / `-e HF_TOKEN`
   （**只传变量名**），值留在子进程环境；新增 `redact_command()` +
   `BenchCommand.display()`，`run` 和 `check` 打印的命令都会脱敏。
   测试：`test_api_key_never_reaches_the_argv`、`test_redact_command_hides_secret_values`、
   `test_display_is_redacted`。
5. **容器以 root 运行**。新增 `runner.docker_user`（默认 `auto`）：Linux 上加
   `--user $(id -u):$(id -g)`，macOS 上是 no-op（Docker Desktop/OrbStack 已做映射）。
   测试：`test_docker_user_auto_only_applies_on_linux`、`test_docker_user_appears_on_linux`。
6. **超时抛原始 `TimeoutExpired` 且容器继续跑**。改用 `Popen` + `threading.Timer` 看门狗：
   超时先 `docker kill <--name>` 再 `proc.kill()`，最后抛 `RunnerError`。
   测试：`test_timeout_is_reported_as_runner_error`（真起一个 `sleep 30` 并断言容器被 kill）。
7. **YAML 里写数字会被拒**。`request_rate` 加 before-validator（数字→字符串，
   并校验 >0 或 `inf`）；`metadata` 的值统一 stringify。
   测试：`test_numeric_request_rate_from_yaml`、`test_bad_request_rate_is_rejected`、
   `test_non_string_metadata_values`。
8. **拼错字段名会被静默忽略**。所有 section 继承 `_Section`（`extra="forbid"`），
   `AppConfig` 同样 forbid，并在 `load_config` 里显式检查顶层 key。
   测试：`test_typo_in_a_section_field_is_rejected`、`test_unknown_top_level_key_is_rejected`。
9. **失败请求上报 0ms 延迟，拉低平台上的均值**。`RequestMetrics.feedback_scores()` 只在
   `succeeded` 时才发 `ttft_ms`/`tpot_ms`/`e2e_ms`/`output_tokens_per_s`；
   `success=0` 和 token 计数照发，失败仍然可见。
   测试：`test_failed_request_publishes_no_zero_latencies`、
   `test_failed_request_trace_has_no_zero_latency_scores`。
10. **有失败也退出 0**。新增 `cli.failure_summary()`：`completed < num_prompts`
    或存在 error 时打印 WARN（含前 3 条**去重后**的错误原文）并以 1 退出；
    `--allow-failures` 可覆盖。结果**仍然**照常同步。
    测试：`tests/test_cli.py` 共 11 个用例，含 `run` 的退出码接线。
11. **`check` 把 0 个模型当成 OK**；`base_url` 带 `/v1` 不会被处理。
    前者改为 FAIL；后者加 `_normalise_base_url` 校验器自动剥掉尾部 `/v1`。
    测试：`test_base_url_with_v1_suffix_is_stripped`。
12. **`sample_id` 是纯位置的**（`sample-0004`），换了内容重新生成 JSONL 会在同一个平台
    dataset 里出现重号，映射会取到任意/过期的那条。改成
    `s{行号:04d}-{sha1(prompt)[:12]}`；同时 `index_dataset_items()` 遇到重复 `sample_id`
    直接抛 `SyncError` 而不是悄悄覆盖。
    测试：`test_sample_id_is_content_derived`、`test_sample_id_disambiguates_duplicate_prompts`、
    `test_duplicate_sample_ids_are_refused`。
    ⚠️ 这会改变 id 方案，所以本机上把旧的 `vllm-bench-samples` dataset 删掉重建了。
13. **`--ready-check-timeout-sec` 在 vLLM < 0.10.2 不存在**。extra 升到 `vllm>=0.10.2`，
    并把 `server.ready_check_timeout_sec` 改成 `Optional[int]`：设为 `null` 就不下发该参数。
    测试：`test_ready_check_flag_can_be_omitted_for_old_vllm`。
14. **`uv.lock` 里全是清华镜像地址**（来自本机 `~/.config/uv/uv.toml`）。
    只用 `UV_DEFAULT_INDEX=... uv lock` 重新生成是不够的——本机再跑一次 `uv sync`
    就会把镜像地址写回去。最终在 `pyproject.toml` 里加了
    `[[tool.uv.index]] url = "https://pypi.org/simple", default = true`，
    让锁文件对机器配置免疫。现在 `uv.lock` 里 0 处镜像地址、4841 处
    `files.pythonhosted.org`，且 `uv lock --check` 在本机默认配置下也通过。
    用户仍可用 `UV_DEFAULT_INDEX=<url> uv sync` 走自己的镜像。
15. **默认配置是照着这台 Mac 调的**（docker + linux/arm64）。
    `config.py` 与 `config.example.yaml` 的默认改成 `mode: native` + `docker_platform: null`，
    docker 那套作为「本机没有 vLLM」的备选方案以注释形式保留。
    本机的 `config.yaml` 仍然是 docker 模式。
    测试：`test_example_config_is_valid_and_portable`。

### NICE-TO-HAVE

16. 环境变量里的 `""` / `null` / `none` / `~` 现在会把 Optional 字段置为 `None`。
    测试：`test_env_null_clears_an_optional_field`。
17. 聚合 feedback score 改为遍历结果里**实际存在**的分位数，
    `--metric-percentiles 75,99.9` 也能同步。测试：`test_custom_percentiles_are_all_published`。
18. `check` 不再用 `get_or_create_dataset` 创建 dataset，改成只读探测
    （`dataset_exists()`），不存在就打印 INFO。
19. 压测输出改为**实时流式**打印（`Popen` + 逐行转发，stderr 合并进 stdout 保持时序），
    不再等进程结束才一次性 dump。`RunOutcome.stdout/stderr` 合并成 `RunOutcome.output`。
20. 文档：`platform_sync` docstring 的 `source` 改成 `dataset_source`；README 补充
    `cd` 的目录说明、`server.tokenizer` 可以填本地目录、docker 模式在 Linux 上要求被测服务
    监听 `0.0.0.0`（并给出 `--network host` 的替代方案）、`benchmark_platform.url` 默认端口 5173 的说明、
    以及索引钉定的说明。
21. 非 `custom` 数据集 + 平台上不存在该 Dataset 时，`sync_result` 抛 `SyncError`，
    不再悄悄建一个空 dataset + 没有 item 的 experiment；CLI 捕获后以 1 退出。
    测试：`test_sync_refuses_to_create_an_empty_dataset`、
    `test_sync_without_samples_is_allowed_against_an_existing_dataset`。

### 修复后的回归验证（本机）

* `uv run pytest` → **87 passed**（原来 43）。
* `uv lock --check` 在本机默认（镜像）配置下通过，`uv.lock` 无镜像地址。
* docker 模式 3-prompt 重跑：`results/vllm-bench-qwable-9b-claude-fable-5-20260917-174941.json`，
  3/3 completed；duration 12.67s / mean TTFT 137.83ms / mean TPOT 51.74ms /
  mean E2EL 6414.58ms / 输出 TPS 28.98 / 总 TPS 48.24。
* 平台（`vllm-bench-samples` dataset 3 items、experiment `vllm-bench-postfix-smoke`
  trace_count=3、project `vllm-bench` 4 条 trace = 3 request + 1 summary(31 scores)），
  dataset item 的新 id 形如 `s0000-73a6cb0a2bde`。
* 对齐在 n=3 时是非平凡置换 `[0, 2, 1]`，request-0001 正确指向 line 2。
* 退出码：干净运行 `EXIT=0`；被拒绝的 sync（item 21）`EXIT=1` 且**没有**建出空 dataset。


---

## 9. 对外命名：一律称「Benchmark 平台」

内网的压测平台是基于上游开源项目自建的，**对外只叫 Benchmark 平台**。
因此所有用户可见的东西——包名、模块名、测试文件名、命令名、环境变量前缀、
配置小节、帮助文本、日志行、README/NOTES——都只出现 "Benchmark 平台"：

* 包 `vllm_bench_platform/`，同步模块 `platform_sync.py`，测试 `tests/test_platform_sync.py`
* 命令 `vllm-bench-platform`，短别名 `vbp`
* 环境变量前缀 `VBP_`（本节为 `VBP_BENCHMARK_PLATFORM__*`）
* 配置小节 `benchmark_platform:`（url / workspace / api_key / project_name /
  dataset_name / experiment_name 等键名未变）
* 配置类 `PlatformSettings`、客户端工厂 `build_platform_client()`、参数 `platform_cfg`、
  测试 fake `FakePlatformClient`（这些标识符原本就带上游项目名，属于改名本身，
  不是额外重构；其余内部标识符一律没动）
* 容器名前缀 `vllm-bench-platform-*`、`experiment_config.tool = "vllm-bench-platform"`
* `check` 的分节标题 `[4/4] Benchmark 平台`、SDK 版本行 `platform SDK ok (v1.11.14)`

Docker 镜像名 `vllm-bench-client:0.11.0`、结果文件名 `vllm-bench-<model>-<ts>.json`、
平台上的 project `vllm-bench` / dataset `vllm-bench-samples` / experiment
`vllm-bench-{model}-{ts}` 本来就不带上游项目名，保持不变。

### SDK 日志前缀也做了改名

SDK 会在自己的 logger 上装一个 `"<上游名>: %(message)s"` 的 StreamHandler，
`run` / `sync` 时会往控制台打一行「Started logging traces to ...」。这行用户直接看得到，
所以 `platform_sync.rebrand_sdk_logging()` 在建客户端时就地把 formatter 前缀换成
`"Benchmark 平台: "`——只改前缀，URL 照常输出；整段包在 try/except 里，
这类纯装饰性的失败绝不影响同步。对应测试：`test_sdk_console_log_prefix_is_rebranded`、
`test_rebrand_never_raises`。

### 刻意保留的地方

平台的 Python SDK 以 `opik` 这个包名发布，所以下面这些**必须**保留原样，
它们是依赖/接口标识，不是产品名：

1. `import opik` / `opik.Opik(...)` / `from opik.api_objects...` —— SDK 包名与类名；
2. `pyproject.toml` 的依赖 `opik>=1.9.8,<2` 及 `[tool.uv.sources]` 注释模板；
3. `config.py` 的 `_ALIASES` 里 `OPIK_URL_OVERRIDE` / `OPIK_WORKSPACE` /
   `OPIK_API_KEY` / `OPIK_PROJECT_NAME` —— 底层 SDK 自带的环境变量回退；
4. `platform_sync.rebrand_sdk_logging()` 里的 logger 名和被替换的旧前缀字面量
   —— 改名逻辑的匹配输入，必须精确；
5. README 的「关于平台 SDK 版本」一段 + 本节，说明 SDK 的包名与
   `~/.opik.config` 不会被读写，否则用户看到依赖名会困惑。
   环境变量清单只在 README「配置」一节列一次（标题即「底层 SDK 环境变量兼容」），
   `config.example.yaml` 只指向 README，不再重复枚举。


---

## 10. Trace 可用性改造（per-request capture + spans）

**问题**：平台上记录的 trace 不具备可用性——每条 trace 的 `end_time`/`duration` 都是空、
`start_time` 全部等于"同步那一刻"因而完全相同，时间轴毫无意义；没有 span，看不出
TTFT 阶段和解码阶段；逐请求 metadata 只有几个字段；汇总 trace 平铺 31 个 feedback score
（连 `total_input_tokens` 这种计数器都在里面），且 input 把整个 run_config 又抄了一遍。

**根因**：`vllm bench serve` 的结果 JSON 里根本没有逐请求的 prompt、请求 ID 和墙钟时间戳，
只有几个对齐的数组。光靠它无论如何拼不出时间轴。

### A. 逐请求采集 sidecar

新增 `vllm_bench_platform/vllm_entry.py`（**只依赖标准库 + vllm**，因此能直接在精简的
压测容器里跑）。它包装
`vllm.benchmarks.lib.endpoint_request_func.ASYNC_REQUEST_FUNCS` 中的每个请求函数，
把每次调用的输入与输出、以及墙钟起止时间，追加到 `<结果>.requests.jsonl`，
再照常调用 `vllm.benchmarks.serve.main()`。

几个关键实现点：

* `ASYNC_REQUEST_FUNCS` 是一个模块级 dict，`serve.py` 是 `import` 进来后**在
  `benchmark()` 内部**才按 backend 取值的，所以**原地修改这个 dict 就够了**，
  不需要 monkeypatch `serve` 本身。
* **预热请求的排除**：`serve.benchmark()` 在正式循环前会发一个预热请求
  （`--profile` 时还有一个 profiler 请求）。它们构造 `RequestFuncInput` 时
  **不带 `request_id`**，而每个被计量的请求都带 —— 所以 `request_id is None`
  是一个精确且不依赖版本号的判别条件。采集时打 `is_warmup` 标记、解析时丢弃，
  同步时再核对条数 == `num_prompts`。实测：3 个 prompt 的运行采到 4 条，
  其中 1 条 `is_warmup=True`。
* `RequestFuncOutput.start_time` 在 0.11.0 里**存在但是 `perf_counter()`**，不是墙钟，
  所以入口自己记 `time.time()`。
* sidecar 的写入顺序是**完成顺序**，而 vLLM 的请求 ID 形如 `<prefix><i>`、`i` 就是发出顺序，
  所以解析时按 ID 的数字后缀排序 —— 实测排序后的 `ttfts` 与结果 JSON 的
  `ttfts` 数组逐项相等，确认两者同序。
* docker 模式把本包**只读挂载**到容器 `/opt/vbp` 并设 `PYTHONPATH`，跑同一个入口，
  采集逻辑只有一份，不会和镜像里的副本漂移。镜像里的 `vllm-bench-serve` 保留为
  `capture: false` 的回退入口。
* 新增配置：`runner.capture`（默认 true）、`runner.python`（默认 `sys.executable`）。
  native 模式下那个解释器必须能 `import vllm`，README 里写明了两种满足方式。

### B. 对齐改为按 prompt 原文匹配

有 sidecar 时就知道每个请求真正发出去的 prompt，于是 `samples.align_by_prompt()`
直接按文本精确匹配 —— 不再需要复算 vLLM 的洗牌，也**不再依赖它的实现细节**，
并且对任意数据集类型都成立。采集到但本地 JSONL 里没有的 prompt 会被按内容去重后
补充成新的 Dataset item，所以 `sharegpt` / `random` 这类运行现在也能拿到关联好的
experiment item。复算洗牌的老路径保留为无 sidecar 时的回退。

### C. Trace 结构

`req-{idx:04d}`，真实 `start_time`/`end_time`；input `{prompt, max_tokens, model}`；
output `{generated_text, success, output_tokens}`；tags 带上 model/backend/dataset；
metadata 扩到 20+ 个字段（endpoint、base_url、request_rate、max_concurrency、
sampling_params、`itl_stats`、`itl_ms` 上限 512 条…）；feedback score **从 7 个砍到 5 个**
（token 计数移进 metadata / span usage）。

每条 request trace 下挂 **2 个 span**：`prefill (TTFT)`（发出 → 首 token）和
`decode`（首 token → 末 token），都是 `type="llm"` 并带 `model`/`provider`，
`decode` 上带 `usage={prompt_tokens, completion_tokens, total_tokens}`——
平台会把它汇总到 trace 级别，token 列因此能自动填充。

### D. 汇总 Trace 瘦身

时间跨度改为真实压测窗口；input 只放运行配置；output 改成**分组**结构
（ttft/tpot/itl/e2el/throughput/counts）；完整原始结果放 metadata；
feedback score **从 31 个砍到 13 个**头部指标。

### 实测验证（docker 模式，3 prompts，max_concurrency=2）

| 检查项 | 结果 |
| --- | --- |
| trace `start_time` 互不相同 | ✅ `req-0000` 与 `req-0001` 同时开始，`req-0002` 在前者结束后开始——正好体现 `max_concurrency=2` |
| `end_time` / `duration` 非空且 ≈ E2E | ✅ `duration=6340.678ms`，`metadata.e2e_ms=6340.678` |
| 每条 trace 2 个 span | ✅ `prefill (TTFT)` `dur=136.678ms` == `ttft_ms`；`decode` `dur=6204.0ms` |
| span usage 生效 | ✅ trace 级 `usage={prompt_tokens:81, completion_tokens:122, total_tokens:203}` |
| metadata 字段齐全 | ✅ 24 个 key，含 `run_id`/`endpoint`/`base_url`/`itl_stats`/`sampling_params` |
| 汇总 trace 分组输出 | ✅ `ttft_ms`/`tpot_ms`/`itl_ms`/`e2el_ms`/`throughput`/`counts` 六组，13 个 feedback score |
| 逐请求 feedback score | ✅ 只剩 `ttft_ms`/`tpot_ms`/`e2e_ms`/`output_tokens_per_s`/`success` |
| experiment item 关联 | ✅ 3/3，按 prompt 原文对齐 |

---

## 11. 老版本 vLLM 兼容（flag 探测 + 降级）

**问题**：用户内网用本机已装的 vLLM（native）跑，直接崩在
`vllm bench serve: error: unrecognized arguments: --custom-skip-chat-template`。
之前的实现把 0.11.0 的参数集写死了。

### 逐版本核对（下载 PyPI sdist 实际 grep，不是猜）

| 参数 / 字段 | 0.9.0 | 0.9.1 | 0.9.2 | 0.10.0 | 0.10.1 | 0.10.2 | 0.11.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `--dataset-path` | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ | ✓ |
| `--dataset-name custom` | ✗ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ |
| `--backend`（旧名 `--endpoint-type`） | ✗ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ |
| `--save-detailed` | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ | ✓ |
| `--custom-output-len` | ✗ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ |
| `--custom-skip-chat-template` | ✗ | ✗ | **✓** | ✓ | ✓ | ✓ | ✓ |
| `--ready-check-timeout-sec` | ✗ | ✗ | ✗ | ✗ | **✓** | ✓ | ✓ |
| `RequestFuncInput.request_id` | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** | ✓ |
| `RequestFuncOutput.start_time` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| `endpoint_request_func` 路径 | `benchmarks/` | 同左 | 同左 | 同左 | **`benchmarks/lib/`** | 同左 | 同左 |

→ **`--custom-skip-chat-template` 是 0.9.2 引入的**，所以用户的 vLLM 是 **0.9.0 或 0.9.1**。
（顺带修正：之前 pyproject 注释里写的 `--ready-check-timeout-sec` 是 0.10.2 引入，
实际是 **0.10.1**。）

**修这个 bug 时挖出来的两个更深的坑**（都是只丢参数解决不了的）：

1. **`--backend` 在 0.9.2 之前根本不存在**，那时叫 `--endpoint-type`。
   所以不能只是"丢掉"，得**换拼写**——加了 `FLAG_ALIASES`。
   （之前的 grep 在 0.9.0 里匹配到 `--backend` 是因为它出现在 `throughput.py`，
   不是 `serve.py`；用真 parser 一跑就现原形了。）
2. **`--dataset-name` 的可选值也是逐版本变的**：0.9.0 只有 `random`，
   0.9.1 有 `sharegpt/burstgpt/sonnet/random/hf`，**`custom` 要到 0.9.2 才有**。
   而"用自己的 prompt 当测评集"正是本工具的核心用法，所以这是**值级别**的不兼容——
   光过滤参数名，用户下一步就会撞上 `invalid choice: 'custom'`。
   探测时一并读出 `--dataset-name` 的 choices，不匹配就报一条说明清楚的 `RunnerError`。

### 实现

1. **能力探测**（`runner.detect_capabilities`）：每次运行前探一次目标 vLLM 的参数表，
   进程内按 (mode, capture, image, platform, python, vllm_bin) 缓存。
   - `capture: true`：跑 `vllm_entry --vbp-probe`，**一次调用同时拿到版本号和完整参数表**
     （用 `add_cli_args` 建 parser 后遍历 `parser._actions`，比解析 help 文本可靠）。
   - `capture: false`：跑 `--help` 并正则抓 `--flag`；native 下再补一次
     `python -c "import vllm; print(vllm.__version__)"` 拿版本。
   - **探测失败 → `detected=False` → 全部照发**（旧行为）+ 一行 WARN，绝不挡住压测。
2. **只在结果可信时才采信探测**：必须包含全部必需参数才算成功。
   这条是写测试时发现的真 bug —— `docker run` 失败会打出**它自己的** usage，
   里面也有 `--flag`，照单全收就会把真参数全删光。
3. **必需 vs 可选**：必需集（`--backend --base-url --endpoint --model --dataset-name
   --dataset-path --num-prompts --save-result --result-dir --result-filename`，0.9.0 起就有）
   缺失 → `RunnerError` 并带上版本号；可选参数缺失 → 静默去掉 + 一行 WARN 说明后果
   （表在 `OPTIONAL_FLAG_CONSEQUENCES`）。`extra_args` 里用户显式写的参数照发，但也会 WARN。
4. **版本落盘**：`experiment_config.vllm_version`、每条 request trace 的
   `metadata.vllm_version`、汇总 trace 的 metadata 与 input。
5. **采集入口的跨版本适配**（`vllm_entry.py`）：
   - `endpoint_request_func` 两条模块路径都试（0.10.1 挪过位置）。
   - 预热识别：有 `request_id` 字段（≥0.10.2）就用"预热请求没有 request_id"；
     没有就退回"第 0 次调用即预热"（预热在主循环前被 await，必然是第 0 次）。
     `load_sidecar(expected=num_prompts)` 再复核一次：若按猜测剔除后条数不对、
     而全部保留正好等于 `num_prompts`，就撤销猜测。
6. **指标缺失容忍**：老版本没有 e2el 聚合时，解析器 / `grouped()` /
   `headline_feedback_scores()` 全部跳过缺失块而不是报错。

### 验证

* 用 0.9.0 / 0.9.1 / 0.10.0 / 0.10.1 的 **真实 sdist 参数集**驱动 `build_command`，
  逐版本断言该丢的丢、该留的留（`tests/test_capabilities.py` 的参数化矩阵）。
  0.9.1 这一档正好复现用户的场景：`--custom-skip-chat-template` 被丢掉而不是崩。
* arm64 只有 0.10.2+ 有 wheel（0.9.x / 0.10.0 / 0.10.1 都只发 x86_64），
  所以老版本的**真实**容器跑测用 `--platform linux/amd64` 模拟。


---

## 12. 平台展示名中文化

**要求**：上传到平台的指标名要用中文，但行业通用术语（TTFT、TPOT、ITL、P50/P90/P99、
tokens/s）保留英文。

**做法**：新增 `vllm_bench_platform/metric_names.py`，作为**唯一**的展示名来源。
逐请求 feedback score、汇总 feedback score、汇总 trace 的分组 `output`、span 名、
汇总 trace 名，全部经它解析。

**边界很清楚**：只翻译**展示**字符串。机器字段一律保持英文，这样脚本、diff、
看板不会因为改名而碎掉：

* 原始结果 JSON（vLLM 自己的格式，一个字不动）
* `experiment_config` 的 `metrics` / `run`
* 所有 trace / span 的 `metadata` 键（`ttft_ms`、`itl_stats`、`run_id` …）
* 采集 sidecar

`grouped()` 内部仍然用英文键构造，最后一步才 `translate_grouped()`；
`headline_feedback_scores()` 用 `percentile_score_name(stat, metric)` 生成，
所以 `mean → 均值`、`median → 中位数`、而 `p50/p90/p99` 原样保留，
自定义分位数（例如 `p95`）也能自动得到 `ITL P95(ms)` 这样的名字，无需再维护映射表。

### 实测验证（真实 docker 跑 + REST 回读）

* 平台**原样接收**中文名，REST 返回的原始字节里是 `"name":"压测汇总"`，
  **没有被转义成 `\uXXXX`、也没有乱码**。
* **按分数名筛选可用**（这点专门验了，否则中文名就只是好看而不好用）：
  * `feedback_scores.请求成功 = 1` → 3 条 request trace
  * `feedback_scores."TTFT(ms)" > 100` → 2 条（第三条 79.39ms 正确被排除）
  * `feedback_scores."请求完成率" = 1` → 压测汇总
  * `feedback_scores."端到端延迟(ms)" > 1000` → 3 条
  * **名字里带括号时查询语句要加引号**（`feedback_scores."TTFT(ms)"`），
    不带括号的可以不加。这条已写进 README，免得用户踩。

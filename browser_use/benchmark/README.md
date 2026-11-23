# Browser Use 基线测试台

测试台复用 Jev Ultrafast 的页面、110 条基线、批次和归档逻辑。页面以本机 Jev 8810 版本为参考，其前端对应源树 `8306149942e03c2ca9c97de7f645f57d74b79fc5`。用例分组取该页面的目录快照。原文件许可见 `LICENSE`。

从 Browser Use 仓库根目录启动。

```powershell
uv venv --python 3.12
uv pip install -r pyproject.toml --extra benchmark
.venv\Scripts\python.exe -m browser_use.benchmark --port 8824
```

打开 http://127.0.0.1:8824/#baseline。自由任务和基线测试均由 Browser Use 执行。浏览器默认使用独立临时配置。批次保存在当前目录的 `artifacts/baselines`，可用 `--artifacts` 指定目录。

默认使用调试配置，复用原 Jev 的 Planner 服务执行、Text 服务独立终验。最终报告沿用原来的模型、服务和专用凭据，仅显式选择时用于当次运行。可用 `--profile-env` 读取已有的本机 Jev 配置，也可以在仓库的 `.env` 配置相同变量。若真实用例需要登录，可用 `BROWSER_USE_BENCHMARK_USER_DATA_DIR` 指定专用 Chrome 配置目录。

新运行支持直接执行。旧归档中的 Planner、重规划、资料库和模型调用详情仍由原版界面呈现。

公开用例 79、83 和 91 使用测试邮箱或测试收货信息。实际跑这三条用例时，可以将用例数据替换为专用测试账号的值。

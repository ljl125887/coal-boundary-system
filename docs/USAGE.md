# 使用与扩展

先安装依赖并执行 README 中的合成示例。两次运行的成果目录相互独立，第二次应输出两条缓存命中信息。演示脚本会检查识别到两个合成堆及共享缓存命中。

## 接入自有数据

复制 `config.example.yaml` 为被忽略的 `config.local.yaml`，设置自己有授权的数据目录和输出目录。必须确认坐标系符合代码支持范围，使用自己的地面/分割参数，按需配置 ROI。ROI、台账、权重和运行成果仅保留在本地。

启用模型前安装可选依赖并提供兼容权重：`python -m pip install -e ".[ml]"`。如果使用参考台账评估，需同时提供适配的表结构和真实参考总体积；默认值 `1.0` 只是占位，不能作为验收依据。

## Worker

与兼容后端联调时设置 `worker.enabled`、`integration.enabled`、后端 URL、输入/输出逻辑根及 `worker.cache_root`。鉴权由环境变量传入，避免写入 YAML 或代码。随后运行：

```bash
python -m coal_boundary.cli --config config.local.yaml worker-once
python -m coal_boundary.cli --config config.local.yaml worker
```

公开示例不包含后端部署、内网拓扑或账号。完整协议行为可阅读 `worker/client.py`、`worker/service.py`、`integration/client.py` 和模拟服务测试。

## 验证

```bash
python -m pytest -q
python -m examples.synthetic_demo
python tools/check_public_release.py
```

模型测试默认跳过；仅当本地显式设置 `COAL_TEST_MODEL_PATH` 时加载指定权重。部分平台的软链接权限测试可能跳过，GUI 交互体验仍需在实际桌面环境验证。

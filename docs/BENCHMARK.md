# 性能测量口径

报告同时记录每帧样本、mean、p50、p95，以及 checkpoint/源码 SHA256、固定网络输入、捕获到的真实帧尺寸、设备与软件版本。

| 字段 | 含义 |
|---|---|
| `read_ms` | 从 OpenCV 获取一帧；静态图像模式为已解码图像的内存拷贝 |
| `preprocess_ms` | BGR→RGB、resize、ImageNet 归一化；PyTorch 模式还包含上传和 dtype 转换 |
| `forward_ms` | 网络前向；PyTorch CUDA 会同步计时；ORT 是一次 Session.run，可能包括 provider 内部拷贝 |
| `transfer_ms` | PyTorch 输出转换为 CPU FP32 NumPy；ORT 此项为 0，因为 Session.run 已返回 NumPy |
| `predict_ms` | 预处理、前向和输出转换的总时间 |
| `postprocess_ms` | 固定尺度解码，以及在需要时进行的可视化 |
| `output_ui_ms` | 请求的 PNG/AVI 输出、imshow 与 waitKey |
| `pipeline_ms` | 本帧 read 开始到上述输出/UI 返回；含 Python 调用间隙 |
| `pipeline_fps` | 已测帧数 / pipeline_ms 的累计秒数；不是逐帧 FPS 的算术均值 |

模型初始化、warmup、报告写入以及最后一帧 NPZ 保存不计入帧耗时。Warmup 使用固定网络形状的合成黑图，不消耗相机或视频帧。图像模式的磁盘解码发生在初始化，不是相机端到端延迟。

GPU 内存只记录 PyTorch allocator 的 allocated/reserved 峰值（含模型、warmup 与测量），不等于设备总显存或 Jetson 统一内存。CPU/ORT 的内存项为 null，不能当作零占用。

## 可重放的本机测量

```bash
python scripts/make_fixture.py --output-dir runs/fixture
python work.py --dualtac-root /path/to/DualTacV2 --source runs/fixture/synthetic.avi --headless --device cpu --warmup 5 --report runs/cpu.json
python work.py --dualtac-root /path/to/DualTacV2 --source runs/fixture/synthetic.avi --headless --device cuda --warmup 5 --report runs/cuda_fp32.json
python work.py --dualtac-root /path/to/DualTacV2 --source runs/fixture/synthetic.avi --headless --device cuda --precision fp16 --warmup 5 --report runs/cuda_fp16.json
python work.py --backend onnx --onnx-model artifacts/dualtac_v6_fp32.onnx --source runs/fixture/synthetic.avi --headless --warmup 5 --report runs/ort_cpu.json
```

生成的输入为确定性彩色渐变、几何图形及平移，共 30 帧，没有深度真值。它只用于检查运行链路与计时；不能据此判断真实图像深度精度。导出的 ONNX `--verify` 也是两个合成张量上的数值一致性检查。

公开的本机验收摘要位于 [VALIDATION.md](VALIDATION.md)。桌面 GPU 的吞吐不是 B01 吞吐。之前 DualTac 的锁定精度属于原模型的 GraspNet 派生测试域，不是本仓库新增的头戴实拍精度。

# 本机软件验收 2026 年 10 月 17 日

本次使用本地真实 checkpoint，模型统计为 **937,227 参数**，输入为 **1×3×240×320**，输出 **1×2×240×320**。通过 strict state_dict 加载，运行后原权重文件 SHA256 未变。

环境：Windows x86_64，Python 3.11.16，PyTorch 2.11.0+cu128，OpenCV 5.0.0，ONNX Runtime 1.31.0；CUDA 设备为 **NVIDIA GeForce RTX 5090 Laptop GPU**。CPU 测试指定四个线程。使用 scripts/make_fixture.py 生成的 30 帧 640×480 合成 AVI，warmup 为五次，headless、无可视化编码。

**这是一次运行验收，没有重复性门槛，也没有深度真值。** 它不能作为 Jetson Nano B01 性能，也不用于声明 FP16 的精度或稳定加速比。报告中的设备信息及逐帧样本是判断这些数字边界的依据。

| 后端 | 帧数 | forward p50 ms | predict p50 ms | pipeline p95 ms | pipeline FPS | 原始报告 |
|---|---:|---:|---:|---:|---:|---|
| PyTorch CPU FP32 | 30 | 48.38 | 49.30 | 56.33 | 19.54 | [JSON](validation/cpu.json) |
| PyTorch CUDA FP32 | 30 | 8.30 | 9.70 | 17.34 | 78.15 | [JSON](validation/cuda_fp32.json) |
| PyTorch CUDA FP16 | 30 | 6.55 | 7.90 | 15.04 | 88.33 | [JSON](validation/cuda_fp16.json) |
| ONNX Runtime CPU FP32 | 30 | 14.78 | 16.04 | 20.07 | 54.63 | [JSON](validation/ort_cpu.json) |

## ONNX 一致性

固定形状 FP32、opset 13，ONNX checker 通过。在全零与固定随机张量两例中，ORT 与 PyTorch 的最大绝对输出偏差分别为 **2.1457672e-05, 0.0001373291**；按 rtol=1e-3、atol=1e-4 的逐元素判断通过。输出第二通道是未归一化 near logits，不能将该绝对差直接当作米制深度误差。详见 [导出记录](validation/onnx_export.json)。

## 自动化测试

15 项测试通过，包含 RGB 通道顺序、归一化、固定尺度解码、非有限值拒绝、三通道可视化、真实 AVI 读写与 EOF、warmup 不丢源帧、帧数限制、PNG/NPZ 输出、失败报告、CLI 参数和真实 checkpoint 前向与不可变性。另检查了 PyTorch/ORT 完整预处理与解码一致性，以及 ONNX 文件和契约错配拒绝。公开 CI 缺少私有资产时会跳过三项集成测试。

## 未在本次验证的范围

- B01 或 Orin 板端相机、温度、功耗与长时间运行；
- 新相机/头戴场景的实际距离误差和 near threshold 标定；
- TensorRT 引擎与 INT8 量化；
- 桌面 CPU/GPU 反复测试的稳定速度比较。

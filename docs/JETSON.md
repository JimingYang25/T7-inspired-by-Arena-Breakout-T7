# Jetson 部署与验收

原型设备是 **Jetson Nano B01**。它与 **Jetson Orin Nano** 是不同平台，安装包不能混用。NVIDIA 的 [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) 将初代 Nano 列在 JetPack 4.6.x 系列；安装 PyTorch 时按 [NVIDIA 的 Jetson 安装文档](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 匹配自己的 JetPack、Python 与架构。

## 路线一 在已有可用环境中运行 PyTorch

如果你已在 B01 上跑通当前 DualTac 模型，可以直接复用那个环境，先验证 `import torch, cv2` 及 `torch.cuda.is_available()`。不要用普通桌面 PyPI 包覆盖 Jetson 的 GPU 版 PyTorch。

```bash
python3 work.py --dualtac-root /path/to/DualTacV2 --source 0 --device cuda --report runs/b01_fp32.json
```

默认 FP32。只有在该平台上验证了输出与性能后，再选择 `--precision fp16`。模型体积小不自动保证 FP16 更快。

当前 DualTacV2 源码包含较新的 Python 注解。若设备 Python 无法解析它，请使用下述 ONNX 路线，不要把桌面源码兼容性当成板端已经通过。

## 路线二 在开发机导出 在 B01 使用 ONNX

开发机安装匹配本机的 PyTorch，然后执行：

```bash
python -m pip install -r requirements-onnx.txt
python -m t7_vision.export --dualtac-root /path/to/DualTacV2 --output artifacts/dualtac_v6_fp32.onnx --verify
```

导出为固定输入 **1×3×240×320**、FP32、opset 13；输出 **1×2×240×320**。必须把 `.onnx` 和紧邻的 `.onnx.json` 一起复制到目标设备。JSON 记录归一化、深度范围、文件校验值、模型来源校验值和导出一致性结果。

目标设备须安装与其 Python、aarch64、JetPack 匹配的 ONNX Runtime。`requirements-onnx.txt` 是开发机依赖入口，不代表 PyPI 为所有旧 Nano 环境提供 GPU wheel。先打印已安装的 provider：

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
python3 work.py --backend onnx --onnx-model artifacts/dualtac_v6_fp32.onnx --source 0 --provider CPUExecutionProvider --report runs/b01_ort_cpu.json
```

若当前安装明确提供 `CUDAExecutionProvider`，可改用它；请求不存在的 provider 会直接报错，不会悄悄宣称 GPU 加速。Session 的 provider 列表仍不等于每个算子都运行在 GPU 上，需要目标平台 profiling 才能确认。

## TensorRT 的下一步

固定形状 ONNX 可作为目标设备 TensorRT 构建的输入。需要在目标板上按安装的 TensorRT 版本构建 engine，并测量数值偏差、速度和显存。本次仓库没有 TensorRT engine runtime，也没有把桌面 ONNX 导出写成 TensorRT 已部署。不同平台构建的 engine 不应当作通用文件复制。

## 采集验收清单

1. 记录设备完整名称、JetPack/L4T、Python、推理后端、模型 SHA256 与电源模式；若使用 `nvpmodel` 或 `jetson_clocks`，记录实际设置。
2. 检查相机请求分辨率与报告里的 `actual_frame_size_hw` 是否一致，网络始终使用训练形状 240×320。
3. 同一录制视频分别运行 FP32、FP16 或不同后端。每组保留原始 JSON 与至少三次重复，不挑选最快的一次作为结论。
4. 分开记录纯前向、包含预处理与输出拷贝的 prediction，以及包含采集、可视化和输出的 pipeline 延迟。
5. 真实相机在 0.20–1.00 m 范围分桶采集，核查尺度、边界、光照和时间稳定性。近场概率来自训练 head，不是自动获得的置信度或安全告警。

`--headless` 适合 SSH 测试；默认不编码可视化，此时不能与开启显示或录像的 FPS 混用。使用 `--output runs/preview.avi` 会把可视化和编码放进 pipeline 计时。`--output-fps` 只控制回放速率。

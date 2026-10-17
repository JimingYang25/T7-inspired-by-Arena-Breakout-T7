# T7-inspired-by-Arena-Breakout-T7
A T7(thermal imaging) helmet which is implemented by depth-estimation network 

一个用深度估计网络(Depth-Anything-V2)实现的T7(热成像)头盔
<img width="588" height="271" alt="41ede178a8718691f44271fcf4cec85a" src="https://github.com/user-attachments/assets/428cb9ef-4307-4e57-bf65-4038b7e98d3e" /><img width="588" height="271" alt="fd3d980edc4d06dd6d532a10d3f31f59" src="https://github.com/user-attachments/assets/11e36669-ac85-474b-b07b-93cd58d5b2eb" />

# 0.0 介绍 introduction

  成品展示抖音![抖音/douyin](https://img.shields.io/badge/TikTok-000000?style=flat-square&logo=TikTok&logoColor=white)<link>https://v.douyin.com/elDdhLb8Omg/</link>

  另外，如果你对计算机视觉感兴趣的话不妨去这个仓库看看<link>https://github.com/DepthAnything/Depth-Anything-V2<link/>

  即：该项目用到的视觉神经网络

  In addition, if you are interested in computer vision, you might as well take a look at this repository <link>https://github.com/DepthAnything/Depth-Anything-V2<link/>.

That is, the visual neural network used in this project.

  

> @article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

> @inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
} 

# 1.0 Bill of Material(物料清单)

  1 Jetson Nano B01 <link>https://item.jd.com/10095352326438.html?pcdk=okpP8EdI4cQhlfJysD8SUkQ3jXVOLvdcmq9xfuqnAWXciSeswRLTJkD_wBENtSj5.NnCI.sen1</link>
  
  2 12V锂电池 12V lithium battery <link>https://item.jd.com/10205921319856.html?pcdk=iVwF_6NHmg1KKE4wcrhXKg1ltV9r1WqVDn7fks91iQydIV6Uabkbo3ir1fv04Hkc.icfn.3eJH</link>

  3 降压模块 / voltage transformation unit <link>https://item.jd.com/10102320683824.html?pcdk=8UjAr8z0z-nQQm6zcYy4nr7GEpzMHFJg8t6b27kjb_c0fN9sEcXKg9lYf45YT1mM.icfn.3eJH</link>
   
  4 USB摄像头 USB camera <link>https://item.jd.com/10039679882443.html?pcdk=Uw7ZT24hUEn5XpAu-NRn1NA9hulU-WqKM2VChE20va7n0dWEPLIKZ5hiBpjsrN-n.3z6a.aI3x&spmTag=YTAyNDAuYjAwMjQ5My5jMDAwMDQwMjcuMyUyM3NrdV9jYXJkJTQwMTc4NDI2NDIyNTA5NiUyMzE3NTc0OTk1MjI2Mjk2ODI2OTcxNzclMjMyMDg5ODUzMzQy</link>

  5.头部支架 Head Brace<link>https://item.jd.com/10210901961928.html?pcdk=v3mp5eVnQXD98_liORHR61pYQOtd6BkgfrEMfXtyPm7-SuChZPm_TpsZBzffZHM4.NnCI.sen1</link>

  另外你可能需要打孔器，螺丝钉，螺母，用于固定各零件。Aditionally,you might need drill,screw,nut to fix every parts


# 2.0 文件传递和运行

新版软件接入轻量 **DualTacV2 V6 深度模型（937,227 参数）**，网络输入为 **320×240 RGB**，解码范围为 **0.20–1.00 m**。显示颜色代表模型估计的近场深度；新版输出不具有红外测温含义。历史成品展示、原有物料清单和硬件说明保留在本页其他章节。

完整运行与部署说明：[轻量部署指南](docs/DEPLOYMENT.md) · [Jetson Nano B01 环境](docs/JETSON.md) · [性能测量口径](docs/BENCHMARK.md) · [本机验收结果](docs/VALIDATION.md)。

## 2.1 获取代码和准备环境

在电脑或 Jetson 的终端运行：

```bash
git clone https://github.com/JimingYang25/T7-inspired-by-Arena-Breakout-T7.git
cd T7-inspired-by-Arena-Breakout-T7
python3 -m pip install -r requirements.txt
python3 work.py --help
```

PyTorch 请使用与当前设备、Python 和 JetPack 匹配的版本。Jetson Nano B01 与 Orin Nano 的软件包不能混用。无桌面环境可使用 `opencv-python-headless` 替代 `opencv-python`，运行时增加 `--headless`。

当前轻量模型的源码和训练权重由本地 DualTacV2 工程提供，本次仓库不分发这些资产。需要的文件是 `src/depth_encoder_decoder.py` 与 `checkpoints/depth_pretrained.pt`。

## 2.2 使用本地轻量网络运行

Windows 开发机：

```powershell
python work.py --dualtac-root "D:\Desktop\DualTacV2" --source 0 --device cuda --report runs/camera.json
```

Jetson 已有可用的模型与 PyTorch 环境时：

```bash
python3 work.py --dualtac-root /path/to/DualTacV2 --source 0 --device cuda --report runs/camera.json
```

按 `q` 或 `Esc` 退出。使用 `--source example.jpg` 或 `--source example.avi` 可处理图片或视频；`--frames 300` 限定测量帧数。摄像头的 `--width`、`--height` 与模型固定输入尺寸分别记录。

无窗口录像与数值输出示例：

```bash
python3 work.py --dualtac-root /path/to/DualTacV2 --source example.avi --headless --output runs/preview.avi --save-depth runs/last_depth.npz --report runs/video.json
```

原版运行画面（Depth Anything 历史版本，保留展示链接）：

<img width="1039" height="681" alt="b6695460d689a4241c00e39beee48bb2" src="https://github.com/user-attachments/assets/186c5e4d-150d-4638-a2f6-37efb27b567b" />

## 2.3 导出 ONNX 在目标设备推理

先在具备模型资产的开发机导出：

```bash
python -m pip install -r requirements-onnx.txt
python -m t7_vision.export --dualtac-root /path/to/DualTacV2 --output artifacts/dualtac_v6_fp32.onnx --verify
```

将 `.onnx` 与同名 `.onnx.json` 一起复制到目标设备。目标设备需要安装与其环境匹配的 ONNX Runtime，然后运行：

```bash
python3 work.py --backend onnx --onnx-model artifacts/dualtac_v6_fp32.onnx --source 0 --report runs/ort_camera.json
```

默认使用 CPU provider。只有已安装的运行库明确提供 CUDA provider 时，才能选择 `--provider CUDAExecutionProvider`。详细步骤见 [Jetson 部署说明](docs/JETSON.md)。

## 2.4 性能记录与测试

```bash
python -m unittest discover -s tests -v
```

报告保留每帧耗时、mean/p50/p95、pipeline FPS、软件环境和模型校验值。公开 CI 不需要私有模型资产；有本地权重时可启用额外集成测试。

当前 [验收结果](docs/VALIDATION.md) 来自桌面合成视频运行，不代表 B01 的真实相机性能或新场景深度精度。板端延迟、功耗和真实近场标定需单独测量。

# 3.0 补充和提示 finally and hints

这个仓库只有最关键的部分的说明，支架/泛化代码的设计需要各位自由发挥

不会的部分问AI,问AI,问AI！！！！！！

请有嵌入式的基础知识再做这个项目，别冲动消费!!!!!!

如实在需要鄙人帮助，请私信我的抖音哈!!!顺便帮我点点这个仓库的star，非常感谢!!!

This repository only contains documentation for its most critical components; everyone needs to design the bracket and generalization code independently as they see fit.


Ask the AI for help with any parts you don’t understand,ask the AI,ask the AI, ask the AI!!!!!!


Only start this project if you have a basic foundation in embedded systems. Don’t rush into unnecessary purchases!!!!!!

If you really need my help, please send me a private message on Douyin!!! Also, could you hit the star for this repository? Thanks a lot!!!

  

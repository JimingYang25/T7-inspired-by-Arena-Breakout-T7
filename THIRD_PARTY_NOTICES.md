# Third party components and model assets

The historical `depth_anything_v2/` implementation and existing `checkpoints/depth_anything_v2_vits.pth` come from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), whose code and Small model are released under Apache 2.0. Preserve the source headers and the [upstream license](licenses/DepthAnything-APACHE-2.0.txt). These assets predate this deployment refactor and are retained for provenance.

The new default pipeline adapts a user-supplied **local DualTacV2** model through its class and checkpoint contract. Its architecture, checkpoints, exported ONNX files and private Hub access are not distributed by this change. Obtain those assets and their applicable permissions from their owner. The adapter does not change the licensing of those assets.

The head-mounted demonstration is inspired by a game visual style. This project uses RGB-derived depth estimates; it does not provide thermal measurements or reproduce an infrared sensor.

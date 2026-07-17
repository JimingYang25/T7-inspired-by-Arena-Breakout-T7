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
当你按照jetson nano b01的文档完成烧录以及支架的设计和固定后，再考虑进行这一步

打开你本地电脑的终端(cmd）

Proceed to this step only after you finish flashing the system and designing and installing the bracket in accordance with the Jetson Nano B01 documentation.
Open the terminal (cmd) on your local computer.

输入(key in):
```cmd
scp <Windows下载目录>/T7-inspired-by-Arena-Breakout-T7-main.zip jetson的用户名@<Jetson IP地址>:<Jetson目标路径>
```

然后开机jetson nano b01 , 在它的终端里输入(Then power on the Jetson Nano B01 and type commands in its terminal):
```bash
unzip <Jetson目标路径> <解压后目标路径>

cd <解压后目标路径>/T7-inspired-by-Arena-Breakout-T7-main

#在你python前需要安装pytorch,opencv2
python3 ./work.py
```

然后你就可以看到一个深度图像窗口 Then you can see a depth image window

<img width="1039" height="681" alt="b6695460d689a4241c00e39beee48bb2" src="https://github.com/user-attachments/assets/186c5e4d-150d-4638-a2f6-37efb27b567b" />

# 3.0 补充和提示 finally and hints

这个仓库只有最关键的部分的说明，支架/泛化代码的设计需要各位自由发挥

不会的部分问AI,问AI,问AI！！！！！！

请有嵌入式的基础知识再做这个项目，别冲动消费!!!!!!

如实在需要鄙人帮助，请私信我的抖音哈!!!顺便帮我点点这个仓库的star，非常感谢!!!

This repository only contains documentation for its most critical components; everyone needs to design the bracket and generalization code independently as they see fit.


Ask the AI for help with any parts you don’t understand,ask the AI,ask the AI, ask the AI!!!!!!


Only start this project if you have a basic foundation in embedded systems. Don’t rush into unnecessary purchases!!!!!!

If you really need my help, please send me a private message on Douyin!!! Also, could you hit the star for this repository? Thanks a lot!!!

  

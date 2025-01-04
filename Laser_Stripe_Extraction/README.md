# 文件结构

```
  ├── model.py: SSLSE、VSLSE模型搭建
  ├── dataset.py: 数据集读取
  ├── demo.py: 模型效果演示
  ├── loss.py: 损失函数
  ├── evaluate.py: 模型精度测试
  ├── train_SSLSE.py: SSLSE模型训练脚本
  └── train_VSLSE.py: VSLSE模型训练脚本
```

当前版本主要实现了论文第二章与第三章的SSLSE与VSLSE模型

尚未实现内容包括：

* focal loss
  * 参考：https://github.com/clcarwin/focal_loss_pytorch
* Inverted Residual Block
  * 参考：https://github.com/d-li14/mobilenetv2.pytorch
* CGNLBlock
  * 参考：https://github.com/kaiyuyue/cgnl-network.pytorch
* 稀疏化训练与模型裁剪
  * 参考：https://github.com/Eric-mingjie/network-slimming

# 依赖

```
python>=3.6
torch>=1.10.0
torchvision
tensorboard
numpy
pillow
scipy
```

# PyTorch实现的联邦学习基线方法

PyTorch-Federated-Learning 提供了使用PyTorch框架实现的各种联邦学习基线方法。代码库遵循客户端-服务器架构，非常直观和易于上手。

如果你觉得这个仓库有用，请用你的小星星:star:支持一下。非常感谢！

[英文](README.md)|[简体中文](README.zh-CN.md)<br>

* **当前的基线实现**: Pytorch 实现的联邦学习基线。目前支持的基线有 FedAvg、FedNova、FedProx 和 SCAFFOLD
* **数据集预处理**: 自动下载常用的公开数据集，并将其根据联邦学习特点分割给多个客户端，比如按照不同的非独立同分布要求进行分割。目前支持的数据集有 MNIST，Fashion-MNIST，SVHN，CIFAR-10，CIFAR-100。其他数据集需要手动下载。
* **后处理**: 为评估而进行的训练结果可视化。


## 安装

### 依赖项

 - Python (3.8)
 - PyTorch (1.8.1)
 - OpenCV (4.5)
 - numpy (1.21.5)


### 安装要求

运行: `pip install -r requirements.txt` to install the required packages.

## 联邦数据集预处理

此预处理旨在将整个数据集根据联邦设置分配给指定数量的客户端。根据每个本地数据集中的类别数量，整个数据集被划分为非独立同分布(Non-IID)数据集，这是根据标签分布偏斜来决定的。


## 执行联邦学习基线

### 测试运行
在一个 yaml 文件中定义超参数，例如 "./config/test_config.yaml", 然后只需用这个配置运行：

```
python fl_main.py --config "./config/test_config.yaml"
```


## 性能评估

运行 `python postprocessing/eval_main.py -rr 'results'` 以绘制测试精度和训练损失随着轮数或通信轮数的增加。注意，图中的标签是结果文件的名称。

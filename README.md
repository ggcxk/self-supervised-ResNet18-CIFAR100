# self-supervised-ResNet18-CIFAR100
监督学习与自监督学习在CIFAR-100图像分类任务中的表现

自监督学习：使用ResNet18架构，在CIFAR-10上利用SimCLR框架进行无监督学习提取特征，得到预训练模型权重及log文件保存在  https://pan.baidu.com/s/1q2Z_1h29hzkhbYtA0tsYTQ?pwd=4i9c   然后在CIFAR-100上进行linear classification protocol

baseline（监督学习）：使用ResNet18在CIFAR-100上进行图像分类，结果见  https://github.com/ggcxk/cifar100

### 无(自)监督预训练
```
python pre_train.py
```

### 在CIFAR-100上进行linear classification protocol
```
python finetune_test.py
```

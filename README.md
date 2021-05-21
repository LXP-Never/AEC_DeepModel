# AEC_DeepModel
 基于深度学习的声学回声消除基线代码
 
具体解析见我的博客：https://www.cnblogs.com/LXP-Never/p/14779360.html


# 数据准备

  按照以下文件结构，放好语音，我直接使用的是AEC-Challenge 数据集中的合成数据集
    
```angular2html
└─Synthetic
    ├─TEST
    │  ├─echo_signal
    │  ├─farend_speech
    │  ├─nearend_mic_signal
    │  └─nearend_speech
    ├─TRAIN
    │  ├─echo_signal
    │  ├─farend_speech
    │  ├─nearend_mic_signal
    │  └─nearend_speech
    └─VAL
        ├─echo_signal
        ├─farend_speech
        ├─nearend_mic_signal
        └─nearend_speech
```

数据处理脚本为 `data_preparation.py` 

如果想要自己生成回声的话建议使用 [RIR-Generator](https://github.com/ehabets/RIR-Generator) 方法，毕竟很多论文中使用的也是这个方法



# 运行

```
python train.py
```
具体的命令行解析参数见`train.py`脚本

# 估计近端语音

```
python test/model_test.py
```


点赞，关注，不迷路

我以后还会开源更有价值的内容

# 参考

- [语音数据增强及python实现](https://www.cnblogs.com/LXP-Never/p/13404523.html)


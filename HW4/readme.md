# Homework4: Transformer应用 - Math Word Problem

## 任务描述

此次作业主要目标是使用Transformer模型解决数学文字问题（MWP），即输入数学文本，输出求解该问题的方程。

## 任务要求

1. 使用Transformer模型改进基于RNN的MWP模型。
2. 在30个epoch内，测试集上的准确率达到20%以上。

## 作业提交

- 提交地址：[HW4提交](https://bhpan.buaa.edu.cn/link/AADA1AC4BE9F4C475CA2E45A2E145546A4)
- 截止时间：2024-06-12 12:00

## Hint

一个推荐的Transformer超参数如下：

- num_layers: 4
- embedding_dim: 128
- num_heads: 8
- ffn_dim: 512
- dropout: 0.1
- learning_rate: 0.0008

其中学习率可能需要个性化调整。

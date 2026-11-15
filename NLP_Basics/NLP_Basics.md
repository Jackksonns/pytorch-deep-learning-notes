# 经典序列模型与 NLP 小任务

## 1. 模型简介

本目录沿着“词元表示 → 序列建模 → 任务输出”的路线学习 RNN、LSTM、BiLSTM、TextCNN 和 Encoder–Decoder。代码使用少量内置样本实现

## 2. 核心原理

### RNN 与 LSTM

普通 RNN 在每个时间步将当前输入 `x_t` 与前一隐藏状态 `h_(t-1)` 合并，计算 `h_t = tanh(W_x x_t + W_h h_(t-1) + b)`。同一个单元在时间轴上重复使用，输出形状为 `(batch, time, hidden)`。

LSTM 额外维护细胞状态 `c_t`。遗忘门、输入门和输出门控制旧信息保留、新信息写入及当前输出。`rnn_lstm.py` 手写单层单元；任务中的双向、多层 LSTM 使用 `nn.LSTM` 组合，方便看清结构差异。双向模型同时利用前后文，适合分类和标注；自回归生成只能使用已经出现的词元。

### TextCNN 与二维 CNN

TextCNN 先把词元变为向量，再用 `Conv1d` 沿**词序列**提取局部短语特征，经最大池化得到整句表示。它与 [`CNN_Basics`](../经典网络架构/CNN_Basics/CNN.md) 的二维 CNN 不同：后者的卷积核沿图像的**高和宽**滑动。

### 四类任务

| 任务 | 输入与输出 | 演示模型 |
| --- | --- | --- |
| 语言模型 | 前面字符 → 下一字符 | 手写 RNN / LSTM |
| 文本分类 | 整句 → 一个类别 | TextCNN / 双向多层 LSTM |
| 序列标注 | 每个词 → 一个标签 | BiLSTM；填充位置不计入损失 |
| 序列生成 | 输入序列 → 可变长输出序列 | LSTM Encoder–Decoder |

Encoder 将输入压缩为隐藏状态，Decoder 从 `<bos>` 开始逐步预测，在 `<eos>` 结束。训练时使用真实的上一词元作为 Decoder 输入（教师强制）；推理时使用自身的上一预测。复制与反转数字序列只是展示这一流程。

## 3. 环境与运行

从仓库根目录安装 `requirements.txt`，随后运行以下离线示例：

```bash
python NLP_Basics/rnn_lstm.py
python NLP_Basics/language_model.py --demo --cell rnn
python NLP_Basics/language_model.py --demo --cell lstm
python NLP_Basics/text_classification.py --demo --model textcnn
python NLP_Basics/text_classification.py --demo --model bilstm
python NLP_Basics/sequence_labeling.py --demo
python NLP_Basics/seq2seq.py --demo --task reverse
python NLP_Basics/seq2seq.py --demo --task copy
```

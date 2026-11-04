# Mini-GPT：从零实现 Decoder-only Transformer

## 1. 模型简介

之前在 `transformer/MHA.py` 中实现了缩放点积注意力与多头注意力，在 `bert/` 里学习了双向编码器。本实验继续使用 PyTorch 从零实现一个小型 GPT，用**过去的字符预测下一个字符**。

与 BERT 的双向注意力不同，GPT 需要 **Causal Mask（下三角因果掩码）**：第 `t` 个位置不能查看 `t+1` 及之后的内容。

## 2. 核心原理

```text
文字 → 字符 Token ID → Token Embedding + Position Embedding
     → 多层 Pre-LN Transformer Block（Masked Multi-Head Attention + MLP）
     → LayerNorm → Linear → 下一个字符的 Logits
```

对于输入 `x[0:T]`，标签为向后平移一位的 `x[1:T+1]`，优化 CrossEntropyLoss。注意：该教学实现使用**字符级 Tokenizer**，真实大型语言模型通常使用子词 Tokenizer。

- 输入：`(batch_size, seq_len)`。
- Embedding：`(batch_size, seq_len, n_embd)`。
- Q/K/V：`(batch_size, num_heads, seq_len, head_dim)`。
- 预测输出：`(batch_size, seq_len, vocab_size)`。
- Loss：对所有有效位置的下一个 Token 做交叉熵。

## 3. PyTorch 实现

| 文件 | 作用 |
| --- | --- |
| `GPT.py` | `CausalSelfAttention`、`TransformerBlock`、`GPT`；包含张量形状注释 |
| `train.py` | 数据读取、字符字典、90/10 连续划分、随机上下文 Batch、训练与验证 |
| `generate.py` | 读取 checkpoint，支持 Temperature / Top-k 的逐字符生成 |
| `results/` | 运行后保存权重、Loss 图和文本 |

## 4. 环境与数据集

推荐从仓库根目录安装现代模块依赖：

```bash
python -m pip install -r requirements.txt
```

使用 [karpathy/char-rnn Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)（约 110 万字符）。首次下载需联网：

```bash
cd Mini-GPT
python train.py --download --steps 1200 --block-size 128 --batch-size 32
python generate.py --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top-k 20
```

注意：如果提示词使用了训练词表中不存在的字符，会明确报错；可以通过 `--prompt` 修改。

### 不联网的 Smoke Test（仅检查代码）

```bash
python train.py --demo --steps 2 --block-size 16 --embed-dim 32 --heads 4 --layers 2 --batch-size 4 --eval-iters 1
python generate.py --checkpoint results/mini_gpt_demo.pth --tokens 30
```

`--demo` 使用重复的极短文本，**不能衡量真实语言生成能力**。这时生成脚本会使用 demo checkpoint。



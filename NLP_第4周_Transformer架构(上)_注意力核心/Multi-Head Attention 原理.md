# Multi-Head Attention 机制深度解析与公式推导

作为数据挖掘领域的从业者，理解 Transformer 架构中的核心——**多头注意力机制（Multi-Head Attention）**，是掌握现代大语言模型（LLMs）的基础。本文将深入解析 Multi-Head Attention 的原理，并对其进行严谨的公式推导与维度分析。

---

## 1. 核心思想

在标准的自注意力机制（Self-Attention）中，输入的词向量通过一组共享的 $W^Q, W^K, W^V$ 映射到单一的表示子空间。这会导致模型在计算注意力时，往往只能关注到一种特定类型的相关性（例如，可能只关注到了语法结构，或者只关注了距离较近的词语）。

**Multi-Head Attention** 的核心思想在于：**将单一的注意力计算拆分为多个独立的“头（Head）”**。

每个头拥有自己独立的权重矩阵（$W_i^Q, W_i^K, W_i^V$），从而使得模型能够同时在**不同的表示子空间（Representation Subspaces）**中捕捉输入序列的不同维度的特征（例如：头1捕捉主谓关系，头2捕捉指代消解，头3捕捉长距离上下文依赖）。

---

## 2. 公式推导与维度解析

假设我们的输入序列由 $n$ 个词组成，每个词被表示为维度为 $d_{model}$ 的向量。将这 $n$ 个词打包成一个矩阵 $X \in \mathbb{R}^{n \times d_{model}}$。

在 Multi-Head Attention 中，我们需要指定“头”的数量，记为 $h$。为了保证多头注意力的计算量与单头注意力基本一致，我们通常会对每个头的维度进行降维处理：

$$
d_k = d_v = \frac{d_{model}}{h}
$$

其中，$d_k$ 和 $d_v$ 分别是每个头内部的 Key（或 Query）和 Value 的维度。

### 第一步：线性映射

对于第 $i$ 个注意力头（$i = 1, 2, \dots, h$），我们首先通过独立的线性变换（即乘以权重矩阵）将输入映射到当前头的子空间：

$$
Q_i = X W_i^Q
$$

$$
K_i = X W_i^K
$$

$$
V_i = X W_i^V
$$

**维度分析：**

- 输入矩阵：$X \in \mathbb{R}^{n \times d_{model}}$
- 权重矩阵：
  - $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
  - $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
  - $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- 映射后的结果：
  - $Q_i, K_i \in \mathbb{R}^{n \times d_k}$
  - $V_i \in \mathbb{R}^{n \times d_v}$

> *注意：在实际代码（如 PyTorch）实现中，为了提高计算效率，通常不会显式地使用 $h$ 个小矩阵相乘，而是使用一个大的矩阵（例如 $W^Q \in \mathbb{R}^{d_{model} \times d_{model}}$）进行一次计算后，再通过 `reshape` 和 `transpose` 操作将其切分为 $h$ 个头。*

### 第二步：计算缩放点积注意力

对于每一个头 $i$，独立地计算其标准的 Scaled Dot-Product Attention：

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

**推导过程与维度解析：**

1. **点积打分**：$Q_i K_i^T \in \mathbb{R}^{n \times n}$。这一步计算了序列中每两个位置之间的原始相似度得分。
2. **缩放（Scaling）**：除以 $\sqrt{d_k}$。由于维度 $d_k$ 变大时，点积结果的方差也会变大，导致 Softmax 的梯度消失（推向两端），因此需要缩放因子来稳定梯度。
3. **归一化（Softmax）**：沿着最后一个维度进行 Softmax 操作，得到注意力权重矩阵 $A_i \in \mathbb{R}^{n \times n}$，满足每行之和为 1。
4. **加权求和**：$A_i V_i \in \mathbb{R}^{n \times d_v}$。将注意力权重应用于 $V_i$，得到第 $i$ 个头的输出 $\text{head}_i$。

### 第三步：多头拼接

将 $h$ 个头计算得到的特征表示拼接在一起。由于每个 $\text{head}_i \in \mathbb{R}^{n \times d_v}$，拼接后的结果将恢复到原本的隐藏层维度。

$$
\text{Concat}(\text{head}_1, \dots, \text{head}_h) \in \mathbb{R}^{n \times (h \cdot d_v)}
$$

因为我们预先设定了 $d_v = \frac{d_{model}}{h}$，所以 $h \cdot d_v = d_{model}$。

拼接后的矩阵维度为 $\mathbb{R}^{n \times d_{model}}$。

### 第四步：最终线性映射

最后，为了对多头拼接后的特征进行融合和变换，我们会通过一个输出权重矩阵 $W^O$ 进行最后一次线性映射：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

**维度分析：**

- 输出权重矩阵：$W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ （注：这里的 $d_{model}$ 就是 $h \cdot d_v$）
- 最终输出结果：$\text{MultiHead}(Q, K, V) \in \mathbb{R}^{n \times d_{model}}$

---

## 3. Multi-Head Attention 机制公式

将上述步骤综合起来，Multi-Head Attention 的完整数学表达式为：

$$
\begin{align*}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
\text{where } \text{head}_i &= \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\end{align*}
$$

---

## 4. Multi-Head Attention 的优势

1. **联合不同子空间的表示（Joint Information from Different Subspaces）**
   单头注意力相当于只有一个“观察视角”。多头注意力赋予了模型多个“观察视角”，有的头专注于捕捉局部短距离的句法结构，有的头专注于捕捉全局长距离的语义关联。最终通过 $W^O$ 将这些多元化的信息融合。

2. **防止模型过度关注局部（类似 CNN 的多通道机制）**
   在某种程度上，Multi-Head Attention 类似于卷积神经网络（CNN）中的多通道（Multiple Channels / Feature Maps）。不同的头提取了不同的特征图谱，增加了模型的表达能力和鲁棒性。

3. **并行计算优势**
   虽然分成了多个头，但由于矩阵乘法的结合律和张量的重塑（Reshape）操作，多个头的计算完全可以在 GPU 上**高度并行化**执行。整体计算复杂度并没有随着头的数量增加而显著提升。

---

## 5. 代码 Demo 实例解析

为了更直观地理解，我们以 `Multi_Head_Attention_Demo.py` 中客服对话文本的特征提取为例，展示 Multi-Head Attention 在代码（如 PyTorch）中的实际张量维度变化过程。

**参数设置：**

- **Batch Size ($B$)**: 1
- **序列长度 ($N$)**: 假设为 $N$ (输入文本的字符数) 151
- **词向量维度 ($d_{model}$)**: 16
- **注意力头数 ($h$)**: 4
- **每个头的维度 ($d_k = d_v$)**: $\frac{d_{model}}{h} = \frac{16}{4} = 4$

### 步骤 0：输入数据准备

文本经过分词和 Embedding 层后，转化为词向量张量 $X$。

- **输入 $X$ 维度**: $\mathbb{R}^{B \times N \times d_{model}} = \mathbb{R}^{1 \times N \times 16}$

### 步骤 1：线性变换与多头切分 

在实际代码中，为了并行计算效率，我们**不会**单独乘以 $h$ 个小矩阵，而是先乘以一个完整的大矩阵，再进行切分：

1. **统一线性映射**：

$$
Q_{all} = X W^Q
$$

> 其中 $W^Q \in \mathbb{R}^{16 \times 16}$。映射后 $Q_{all} \in \mathbb{R}^{1 \times N \times 16}$（$K_{all}$、$V_{all}$ 同理）。

2. **多头切分 (View & Transpose)**：
   
   将最后的一个维度 $d_{model}$ 切分为 $(h, d_k)$，即从 `[1, N, 16]` 重塑为 `[1, N, 4, 4]`。
   为了后续矩阵相乘（在最后两维发生），需要将头数 $h$ 的维度提到前面，进行转置（Transpose）：
   
   - **切分与转置后 $Q, K, V$ 维度**: $\mathbb{R}^{B \times h \times N \times d_k} = \mathbb{R}^{1 \times 4 \times N \times 4}$

### 步骤 2：计算缩放点积注意力 

现在，各个头的计算可以通过高维张量乘法（Batch Matrix Multiplication）一次性完成：

1. **点积打分 (Scores)**：
   
$$
S = Q K^T
$$
   
> 计算 $Q \in \mathbb{R}^{1 \times 4 \times N \times 4}$ 与 $K^T \in \mathbb{R}^{1 \times 4 \times 4 \times N}$ 的乘积。
> - **$S$ 维度**: $\mathbb{R}^{1 \times 4 \times N \times N}$

2. **缩放与 Softmax (Attention Weights)**：
   
   将 $S$ 除以 $\sqrt{d_k} = \sqrt{4} = 2$，并在最后一个维度 $N$ 上执行 Softmax。
   
   - **注意力权重矩阵 $A$ 维度**: $\mathbb{R}^{1 \times 4 \times N \times N}$（即 4 个头各自的 $N \times N$ 注意力分布）。

3. **加权求和 (Context)**：
   
$$
Context = A V
$$
   
> 计算 $A \in \mathbb{R}^{1 \times 4 \times N \times N}$ 与 $V \in \mathbb{R}^{1 \times 4 \times N \times 4}$ 的乘积。
> - **$Context$ 维度**: $\mathbb{R}^{1 \times 4 \times N \times 4}$

### 步骤 3：多头拼接

将各个头的上下文表示重新拼接起来：

1. **维度还原 (Transpose & View)**：
   
   将头数 $h$ 还原回原来的位置：`[1, 4, N, 4]` $\rightarrow$ `[1, N, 4, 4]`。
   将最后两维展平（View）：`[1, N, 4, 4]` $\rightarrow$ `[1, N, 16]`。
   
   - **拼接后张量维度**: $\mathbb{R}^{1 \times N \times 16}$

### 步骤 4：最终线性映射 

通过输出矩阵 $W^O$ 对拼接后的特征做最后一次融合：

$$
Output = Context \cdot W^O
$$

> 其中 $W^O \in \mathbb{R}^{16 \times 16}$。
> - **最终 $Output$ 维度**: $\mathbb{R}^{1 \times N \times 16}$（恢复了与输入 $X$ 完全相同的形状，方便后续送入前馈神经网络层或其他结构）。
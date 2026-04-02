import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Multi-Head Attention 机制代码演示示例 (结合客服对话文本)
# 对应 Markdown 文档中的公式推导步骤
# ==========================================

# 0. 准备输入数据 (客服对话文本)
text = "哎，喂，你好好喂，哎你好，女士，这边是抖音平台客服中心的。是这样的，今天可以来电的话，是通知一下您啊，您在上个月在抖音平台上开开通的抖音直播会员到期了，要给您扣费一个月800，一年的话是9600，问一下您还要不要继续保留使用呢？我说您在上个月在抖音平台上开通的直播会员到期了。问一下您还要不要保留使用。"

# 简单起见，我们进行字符级(Character-level)的Token化
tokens = list(text)
vocab = {char: i for i, char in enumerate(set(tokens))}
vocab_size = len(vocab)
indices = [vocab[char] for char in tokens]

# 将输入转化为张量，添加 batch 维度
input_tensor = torch.tensor([indices]) # [1, seq_len]

# 超参数设置
batch_size = 1
seq_len = len(tokens)
d_model = 16           # 词向量维度 d_model (每个字用 16 维向量表示)
num_heads = 4          # 头数 h
d_k = d_model // num_heads # K 和 Q 的维度: d_k = d_model / h = 4
d_v = d_model // num_heads # V 的维度: d_v = d_model / h = 4

print(f"--- 0. 数据准备 ---")
print(f"输入文本长度 (seq_len): {seq_len}")
print(f"词表大小 (vocab_size): {vocab_size}")
print(f"模型维度 (d_model): {d_model}")
print(f"注意力头数 (num_heads): {num_heads}")
print(f"每个头的维度 (d_k = d_v): {d_k}")

# 使用 Embedding 层将字符索引转换为词向量 X
# X 对应文档中的输入矩阵 X ∈ R^(N x d_model)
embedding = nn.Embedding(vocab_size, d_model)
X = embedding(input_tensor)
print(f"输入矩阵 X 的维度: {X.shape}") # [1, seq_len, d_model]

# ==========================================
# 步骤 1：线性变换生成所有头的 Q, K, V
# 实际实现中，通常先用一个大矩阵映射，然后再切分
# ==========================================
# 注意：nn.Linear(in_features, out_features) 会执行 X @ W^T + b
W_q = nn.Linear(d_model, d_model, bias=False) # 权重矩阵大小 [d_model, d_model]
W_k = nn.Linear(d_model, d_model, bias=False) 
W_v = nn.Linear(d_model, d_model, bias=False) 

Q_all = W_q(X)  # [1, seq_len, d_model]
K_all = W_k(X)  # [1, seq_len, d_model]
V_all = W_v(X)  # [1, seq_len, d_model]

print(f"\n--- 步骤 1：线性变换生成所有头的 Q, K, V ---")
print(f"整体 Q 维度: {Q_all.shape}")

# 将 Q, K, V 拆分为 num_heads 个头
# [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
Q = Q_all.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
K = K_all.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
V = V_all.view(batch_size, seq_len, num_heads, d_v).transpose(1, 2)

print(f"切分后的 Q 维度: {Q.shape} (batch_size, num_heads, seq_len, d_k)")
print(f"切分后的 K 维度: {K.shape}")
print(f"切分后的 V 维度: {V.shape}")

# ==========================================
# 步骤 2：计算缩放点积注意力 (Scaled Dot-Product Attention)
# 公式：Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
# ==========================================
# 计算打分: Q * K^T, 需要转置 K 的最后两维
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
print(f"\n--- 步骤 2：计算缩放点积注意力 ---")
print(f"注意力得分矩阵 (Scores) 维度: {scores.shape} (batch_size, num_heads, seq_len, seq_len)")

# Softmax 归一化
attn_weights = F.softmax(scores, dim=-1)
print(f"注意力权重矩阵 (Weights) 维度: {attn_weights.shape}")

# 验证：每一行的和应该为 1 (针对每个头)
print(f"验证某头某行权重之和 (应为1.0): {attn_weights[0, 0, 0, :].sum().item():.4f}")

# 加权求和
context = torch.matmul(attn_weights, V)
print(f"各头加权求和后输出维度: {context.shape} (batch_size, num_heads, seq_len, d_v)")

# ==========================================
# 步骤 3：多头拼接 (Concat)
# ==========================================
# 将 num_heads 的维度和 seq_len 的维度换回来: [batch_size, seq_len, num_heads, d_v]
context = context.transpose(1, 2).contiguous()
# 拼接所有头: [batch_size, seq_len, num_heads * d_v] 即 [batch_size, seq_len, d_model]
concat_output = context.view(batch_size, seq_len, d_model)

print(f"\n--- 步骤 3：多头拼接 (Concat) ---")
print(f"拼接后的输出维度: {concat_output.shape}")

# ==========================================
# 步骤 4：最终线性映射 (Final Linear Projection)
# 公式：MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
# ==========================================
W_o = nn.Linear(d_model, d_model, bias=False)
output = W_o(concat_output)

print(f"\n--- 步骤 4：最终线性映射 ---")
print(f"最终 Output 维度: {output.shape}")

# ==========================================
# 总结：封装为完整的 PyTorch 模块
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性变换并切分头
        # Q = x @ W_q^T -> shape: [batch_size, seq_len, d_model]
        # -> view: [batch_size, seq_len, num_heads, d_k]
        # -> transpose: [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 3. 多头拼接
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 4. 最终线性映射
        output = self.W_o(context)
        
        return output, attn_weights

# 测试封装好的类
print("\n--- 测试封装好的 MultiHeadAttention 类 ---")
mha_layer = MultiHeadAttention(d_model=16, num_heads=4)
out, attn = mha_layer(X)
print(f"封装调用输出维度: {out.shape}")
print(f"封装调用注意力权重维度: {attn.shape} (batch_size, num_heads, seq_len, seq_len)")

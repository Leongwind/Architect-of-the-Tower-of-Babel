import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Self-Attention 机制代码演示示例 (结合客服对话文本)
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
d_k = 8                # K 和 Q 的维度 (通常 d_k = d_v = d_model / head数)
d_v = 8                # V 的维度

print(f"--- 0. 数据准备 ---")
print(f"输入文本长度 (seq_len): {seq_len}")
print(f"词表大小 (vocab_size): {vocab_size}")

# 使用 Embedding 层将字符索引转换为词向量 X
# X 对应文档中的输入矩阵 X ∈ R^(N x d_model)
# 注意：在实际应用中，这里可能是预训练的词向量或者经过多层网络处理后的特征表示
embedding = nn.Embedding(vocab_size, d_model)
X = embedding(input_tensor)
print(f"输入矩阵 X 的维度: {X.shape}") # [1, seq_len, d_model]

# ==========================================
# 步骤 1：线性变换生成 Q, K, V
# 公式：Q = X * W_q, K = X * W_k, V = X * W_v
# ==========================================
# 我们定义三个线性层（包含可学习的权重矩阵 W^Q, W^K, W^V）
# 注意：nn.Linear(in_features, out_features) 会执行 X @ W^T + b
W_q = nn.Linear(d_model, d_k, bias=False) # 权重矩阵大小 [d_k, d_model]
W_k = nn.Linear(d_model, d_k, bias=False) # 权重矩阵大小 [d_k, d_model]
W_v = nn.Linear(d_model, d_v, bias=False) # 权重矩阵大小 [d_v, d_model]

Q = W_q(X)  # [1, seq_len, d_k]
K = W_k(X)  # [1, seq_len, d_k]
V = W_v(X)  # [1, seq_len, d_v]

print(f"\n--- 步骤 1：线性变换生成 Q, K, V ---")
print(f"Q 维度: {Q.shape}")
print(f"K 维度: {K.shape}")
print(f"V 维度: {V.shape}")

# ==========================================
# 步骤 2：计算注意力得分 (Attention Scores)
# 公式：S = Q * K^T
# ==========================================
# Q 和 K^T 相乘。我们需要对 K 的最后两个维度进行转置
K_T = K.transpose(-2, -1) 

# 点积计算未缩放得分
scores = torch.matmul(Q, K_T) 
print(f"\n--- 步骤 2：计算注意力得分 (Attention Scores) ---")
print(f"Scores (S) 维度: {scores.shape} (即 N x N，这里是 {seq_len} x {seq_len})")

# ==========================================
# 步骤 3：缩放 (Scaling)
# 公式：S_scaled = S / sqrt(d_k)
# ==========================================
scores_scaled = scores / math.sqrt(d_k)
print(f"\n--- 步骤 3：缩放 (Scaling) ---")
print(f"将得分除以 sqrt({d_k}) = {math.sqrt(d_k):.4f}，防止Softmax输入过大导致梯度消失")

# ==========================================
# 步骤 4：Softmax 归一化
# 公式：A = Softmax(S_scaled)
# ==========================================
# 在最后一个维度 (dim=-1, 即序列长度方向) 上进行 softmax
attention_weights = F.softmax(scores_scaled, dim=-1)

print(f"\n--- 步骤 4：Softmax 归一化 ---")
print(f"注意力权重矩阵 A 维度: {attention_weights.shape}")
# 验证：每一行的和应该为 1
print(f"验证某一行权重之和 (应为1.0): {attention_weights[0, 0, :].sum().item():.4f}")

# 找一个具体的字来观察它对其他字的注意力，比如第 13 个字 "女"
target_idx = 13
target_char = tokens[target_idx]
print(f"\n>> 观察示例：第 {target_idx} 个字符 '{target_char}' 对全句的注意力权重分布:")

# 获取该字符对所有字符的注意力权重
char_attention = attention_weights[0, target_idx, :]

# 找出注意力最高的Top 5个字符 (由于模型未训练，这里是随机初始化的权重导致的结果)
top_k = 5
top_values, top_indices = torch.topk(char_attention, top_k)

print(f"'{target_char}' 当前最关注的 {top_k} 个字符是 (未训练状态):")
for i in range(top_k):
    idx = top_indices[i].item()
    val = top_values[i].item()
    print(f"  - 字符 '{tokens[idx]}' (位置 {idx}): 权重 {val:.4f}")

# ==========================================
# 步骤 5：计算加权和 (Weighted Sum)
# 公式：Output = A * V
# ==========================================
output = torch.matmul(attention_weights, V) 

print(f"\n--- 步骤 5：计算加权和 (Weighted Sum) ---")
print(f"最终 Output 维度: {output.shape} (即当前词融合了全局上下文的新表示)")


# ==========================================
# 总结：将上述所有步骤封装为一个完整的 PyTorch 模块
# 公式：Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
# ==========================================
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        # 1. 线性变换
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2 & 3. 点积与缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. 加权和
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# 测试封装好的类
print("\n--- 测试封装好的 SelfAttention 类 ---")
self_attn_layer = SelfAttention(d_model, d_k, d_v)
out, attn = self_attn_layer(X)
print(f"封装调用输出维度: {out.shape}")

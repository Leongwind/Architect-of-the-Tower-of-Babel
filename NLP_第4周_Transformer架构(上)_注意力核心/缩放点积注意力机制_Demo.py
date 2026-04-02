import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 缩放点积注意力机制 (Scaled Dot-Product Attention) 代码演示示例
# 对应 Markdown 文档中的公式推导步骤
# ==========================================

# 0. 准备输入数据 (沿用客服对话文本)
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
d_model = 16  # 词向量维度
d_k = d_model # Key和Query的维度 (这里为了简化，保持与d_model一致)
d_v = d_model # Value的维度

print(f"--- 0. 数据准备 ---")
print(f"输入文本长度 (seq_len): {seq_len}")
print(f"词表大小 (vocab_size): {vocab_size}")
print(f"模型维度 (d_model = d_k = d_v): {d_model}")

# 使用 Embedding 层将字符索引转换为词向量 X
embedding = nn.Embedding(vocab_size, d_model)
X = embedding(input_tensor)
print(f"输入矩阵 X 的维度: {X.shape} (batch_size, seq_len, d_model)")

# 生成 Q, K, V (这里简单起见，使用三个线性层映射)
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

Q = W_q(X) # [1, seq_len, d_k]
K = W_k(X) # [1, seq_len, d_k]
V = W_v(X) # [1, seq_len, d_v]

print(f"\n--- 准备 Q, K, V ---")
print(f"Query 矩阵维度: {Q.shape}")
print(f"Key 矩阵维度: {K.shape}")
print(f"Value 矩阵维度: {V.shape}")

# ==========================================
# 步骤 1：计算注意力分数 (Dot Product)
# 公式：Scores = Q K^T
# ==========================================
# K.transpose(-2, -1) 将最后两个维度倒置，即从 [1, seq_len, d_k] 变为 [1, d_k, seq_len]
scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"\n--- 步骤 1：计算注意力分数 ---")
print(f"未缩放的注意力得分矩阵 (Scores) 维度: {scores.shape} (batch_size, seq_len, seq_len)")

# ==========================================
# 步骤 2：缩放 (Scaling)
# 目的：防止点积结果过大导致 softmax 梯度消失
# 公式：Scaled Scores = Q K^T / sqrt(d_k)
# ==========================================
scaled_scores = scores / math.sqrt(d_k)
print(f"\n--- 步骤 2：缩放 ---")
print(f"缩放后的得分矩阵维度: {scaled_scores.shape}")
print(f"缩放因子 sqrt(d_k): {math.sqrt(d_k):.4f}")

# 可选：掩码 (Masking) - 对应 Markdown 第4部分
# 这里演示一个 Look-ahead Mask (防止看到未来信息，常用于Decoder)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() # 上三角矩阵
scaled_scores_masked = scaled_scores.masked_fill(mask, -1e9)
print(f"\n--- 可选步骤：掩码 (Masking) ---")
print(f"应用掩码后的得分矩阵，右上角元素变为 -1e9")

# ==========================================
# 步骤 3：归一化 (Softmax)
# 公式：A = softmax(Scaled Scores)
# ==========================================
attn_weights = F.softmax(scaled_scores, dim=-1)
attn_weights_masked = F.softmax(scaled_scores_masked, dim=-1)

print(f"\n--- 步骤 3：归一化 ---")
print(f"注意力权重矩阵 (Weights) 维度: {attn_weights.shape}")
print(f"验证某行权重之和 (应为1.0): {attn_weights[0, 0, :].sum().item():.4f}")

# ==========================================
# 步骤 4：加权求和
# 公式：Output = A V
# ==========================================
output = torch.matmul(attn_weights, V)
output_masked = torch.matmul(attn_weights_masked, V)

print(f"\n--- 步骤 4：加权求和 ---")
print(f"最终输出 (Output) 维度: {output.shape} (batch_size, seq_len, d_v)")

# ==========================================
# 总结：封装为完整的 PyTorch 模块
# ==========================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # 1 & 2: 打分与缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 可选：掩码
        if mask is not None:
            # mask 中为 1/True 的地方代表需要遮盖的位置
            scores = scores.masked_fill(mask, -1e9)
            
        # 3: 归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4: 加权求和
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

# 测试封装好的类
print("\n--- 测试封装好的 ScaledDotProductAttention 类 ---")
attention_layer = ScaledDotProductAttention(d_k=d_k)
out, attn = attention_layer(Q, K, V, mask=mask) # 使用了掩码
print(f"封装调用输出维度: {out.shape}")
print(f"封装调用注意力权重维度: {attn.shape} (batch_size, seq_len, seq_len)")

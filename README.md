# NLP理论深度学习小组规划方案（三人版）

**制定日期**：

**版本号**：v1.0

**执行日期**：2026年03月06日 - 2026年5月27日（每周三晚）

**制定者**：学习小组全体成员




## 📋 目录

- [一、运作机制](#一运作机制)
  - [角色分工（每周轮换）](#角色分工每周轮换)
  - [时间管理](#时间管理)
- [二、学习规划（12周）](#二学习规划12周)
  - [第一阶段：基石构建](#第一阶段基石构建第1-3周)
  - [第二阶段：Transformer风暴](#第二阶段transformer风暴第4-7周)
  - [第三阶段：大模型时代](#第三阶段大模型时代第8-12周)
- [三、深度保障建议](#三深度保障建议)
- [四、学习资源推荐](#四学习资源推荐)

---

## 一、运作机制

### 角色分工（每周轮换）

| 角色           | 职责         | 具体任务                                                     |
| -------------- | ------------ | ------------------------------------------------------------ |
| **A - 主讲人** | 知识架构师   | • 准备会议材料（讲义/白板草图/核心问题）<br>• **核心理论精讲**与公式推导<br>• 会议前半段主导讲解（确保逻辑自洽） |
| **B - 挑战者** | 批判性思维官 | • 预习时深入挖掘，记录疑难点<br>• **对比阅读**：查找与本周主题相关的**变体或对立观点**论文/博客<br>•  提出"尖锐"问题，挑战主讲人的假设，激发辩论 |
| **C - 连接者** | 工程与实践官 | • **寻找最小代码实现**（从开源库或论文源码中提取）<br>• 关联工业界应用案例，分析利弊<br>• 整理**会议纪要**与下周计划，维护知识库 |

> **💡 示例：** 第1周张三是主讲人、李四是挑战者、王五是连接者；第2周李四主讲、王五挑战、张三连接，以此类推。
>
> **💡 注意：**挑战者的任务为“对比阅读”，这能极大地拓宽视野，避免陷入单一视角。连接者的任务明确了“最小代码实现”，降低上手门槛。

---

### 时间管理

**单次会议时长：1.5小时**

| 时间段       | 时长   | 内容                                   | 负责人        |
| :----------- | :----- | :------------------------------------- | :------------ |
| **预热环节** | 5 min  | 回顾上周核心结论，介绍本周背景         | 连接者        |
| **讲解环节** | 40 min | 核心理论精讲、公式推导、架构图绘制     | 主讲人        |
| **讨论环节** | 35 min | 挑战者发起提问、全员辩论、深度剖析概念 | 挑战者 + 全员 |
| **实战环节** | 10 min | 代码演示、核心API解读、本周小结        | 连接者        |

---

**💡 注意**：增加了5分钟预热，让每次会议有更好的衔接；微调了各环节时长，让讨论更充分。

## 二、学习规划（12周）

### 第一阶段：基石构建（第1-3周）

#### 第1周：词向量与语言模型基石

**📚 核心内容**

- Word2Vec（Skip-gram / CBOW）
- GloVe
- 统计语言模型基础

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 推导 Skip-gram 损失函数<br/>• 讲解负采样原理<br/>• 对比 Word2Vec 与 GloVe 的本质异同 |
| **挑战者** | • 提问：GloVe 如何结合全局统计信息？<br/>• 提问：Word2Vec 如何处理多义词问题？<br/>• **进阶**：FastText 在 Word2Vec 基础上做了什么改进？ |
| **连接者** | • 演示 Gensim 训练词向量<br/>• 展示词向量相似度计算实验<br/>• 展示“国王-王后”类比推理示例 |

**📖 推荐资料**
- 论文：*Efficient Estimation of Word Representations in Vector Space* (Word2Vec)
- 论文：*GloVe: Global Vectors for Word Representation*

---

#### 第2周：RNN家族与梯度问题

**📚 核心内容**
- RNN 基础架构
- LSTM 与 GRU
- Seq2Seq 模型
- BPTT 与梯度消失/爆炸

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 画出 RNN 展开图<br>• 推导 BPTT（随时间反向传播）<br>• 解释梯度消失原因 |
| **挑战者** | • 提问：LSTM 门控如何解决梯度消失？<br>• 提问：GRU 相比 LSTM 做了哪些简化？ |
| **连接者** | • PyTorch 实现 Char-RNN 文本生成<br>• 对比 RNN/LSTM/GRU 训练速度 |

**📖 推荐资料**
- 论文：*Long Short-Term Memory* (LSTM原论文)
- 教程：*The Unreasonable Effectiveness of Recurrent Neural Networks*

---

#### 第3周：注意力机制的诞生

**📚 核心内容**
- Seq2Seq + Attention
- Bahdanau Attention（加性注意力）
- Luong Attention（乘性注意力）

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 讲解 Attention 权重计算过程<br>• 解释 $score(h_t, h_s)$ 物理意义 |
| **挑战者** | • 提问：Attention 如何解决长距离依赖？<br>• 提问：计算复杂度是多少？ |
| **连接者** | • 代码对比 Additive vs Dot-Product Attention<br>• 可视化 Attention 权重分布 |

**📖 推荐资料**
- 论文：*Neural Machine Translation by Jointly Learning to Align and Translate*

---

### 第二阶段：Transformer风暴（第4-7周）

#### 第4周：Transformer架构（上）—— 注意力核心

**📚 核心内容**
- Self-Attention 机制
- Multi-Head Attention
- 缩放点积注意力

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | ⚠️ **必须白板手推**：<br>$$Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$<br>• 解释为什么要除以 $\sqrt{d_k}$（从向量分布角度） |
| **挑战者** | • 提问：Multi-Head 机制带来什么增益？<br/>• 提问：头数越多越好吗？（引出信息冗余与计算成本讨论）<br/>• **对比**：Self-Attention 与 Cross-Attention 的区别 |
| **连接者** | • 打印 Transformer 中间层 Attention Map<br/>• 可视化模型关注的区域（利用 `bertviz` 或类似工具） |

**📖 推荐资料**
- 论文：*Attention Is All You Need*
- 博客：*The Illustrated Transformer* (Jay Alammar)

---

#### 第5周：Transformer架构（下）—— 工程细节

**📚 核心内容**
- Positional Encoding（位置编码）
- Layer Normalization
- Residual Connection（残差连接）
- Mask 机制（Padding Mask / Look-ahead Mask）

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 解释正弦余弦位置编码设计思想<br>• 对比 Layer Norm vs Batch Norm |
| **挑战者** | • 提问：为什么 Transformer 能并行计算？<br>• 提问：Decoder 的 Mask 如何起作用？ |
| **连接者** | • 展示 PyTorch `nn.Transformer` 输入输出<br>• 演示 Mask 的实际效果 |

**📖 推荐资料**
- 代码：*The Annotated Transformer* (Harvard NLP)

---

#### 第6周：BERT与预训练范式

**📚 核心内容**
- BERT 架构
- Masked Language Model (MLM)
- Next Sentence Prediction (NSP)
- Fine-tuning 范式

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 讲解 BERT 输入表示（Token + Segment + Position）<br>• 解释 MLM 训练目标 |
| **挑战者** | • 提问：NSP 为什么被认为是多余的？<br>• 提问：[CLS] 位为何能代表句意？ |
| **连接者** | • HuggingFace Transformers 实战<br>• 演示 BERT 文本分类任务  |

**📖 推荐资料**
- 论文：*BERT: Pre-training of Deep Bidirectional Transformers*
- 论文：*RoBERTa: A Robustly Optimized BERT Pretraining Approach*

---

#### 第7周：GPT与自回归生成

**📚 核心内容**
- GPT 系列演进（GPT-1 / GPT-2 / GPT-3）
- Autoregressive Language Modeling
- Autoencoding vs Autoregressive 对比

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 对比 BERT（双向）与 GPT（单向）本质区别<br>• 讲解自回归生成原理 |
| **挑战者** | • 提问：GPT-2 到 GPT-3 最大跨越是什么？<br>• 提问：单向模型的局限性？ |
| **连接者** | • GPT-2 文本生成 Demo<br>• 对比不同解码策略（Greedy / Beam / Sampling） |

**📖 推荐资料**
- 论文：*Improving Language Understanding by Generative Pre-Training* (GPT-1)
- 论文：*Language Models are Few-Shot Learners* (GPT-3)

---

### 第三阶段：大模型时代（第8-12周）

#### 第8周：Prompting与In-Context Learning

**📚 核心内容**
- Prompt Engineering
- Zero-shot / Few-shot Learning
- Chain-of-Thought (CoT) Prompting
- In-Context Learning (ICL)

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 解释 Prompt Tuning 原理<br>• 讲解 CoT 如何激发推理能力     |
| **挑战者** | • 提问：ICL 真的没有更新梯度吗？<br>• 提问：Prompt 设计的最佳实践？ |
| **连接者** | • 实测 OpenAI API / 开源模型<br>• 对比不同 Prompt 对结果的影响 |

**📖 推荐资料**
- 论文：*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*

---

#### 第9周：指令微调与RLHF

**📚 核心内容**
- Supervised Fine-Tuning (SFT)
- Reward Model（奖励模型）
- RLHF 流程
- PPO 算法

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 画出 RLHF 三阶段流程图<br>• 讲解 PPO 算法核心思想          |
| **挑战者** | • 提问：Reward Model 会有哪些偏见？<br>• 提问：PPO 训练为何容易不稳定？ |
| **连接者** | • 介绍开源微调框架（LLaMA-Factory / DeepSpeed）              |

**📖 推荐资料**
- 论文：*Training language models to follow instructions with human feedback* (InstructGPT)

---

#### 第10周：参数高效微调（PEFT）

**📚 核心内容**
- Adapter
- Prefix-Tuning
- LoRA（Low-Rank Adaptation）
- QLoRA

**👥 任务分工**

| 角色       | 具体任务                                                     |
| ---------- | ------------------------------------------------------------ |
| **主讲人** | • 讲解 LoRA 低秩分解原理：$W = W_0 + BA$<br>• 对比不同 PEFT 方法 |
| **挑战者** | • 提问：秩 $r$ 一般设多少？<br>• 提问：LoRA 为何比全量微调更流行？ |
| **连接者** | • 实战：单卡用 LoRA 微调开源模型<br>• 对比显存占用与推理速度 |

**📖 推荐资料**
- 论文：*LoRA: Low-Rank Adaptation of Large Language Models*

---

#### 第11-12周：实战专题（二选一）

##### 🔹 方向A：检索增强生成（RAG）

**核心内容**
- Embedding Models
- 向量数据库（Faiss / Milvus / Chroma）
- 混合检索
- RAG 架构设计（Naive RAG -> Advanced RAG -> Modular RAG）

**任务**

- 实战：搭建一个简单的**文档问答系统**（如针对公司内部PDF）
- 讨论：RAG 如何缓解幻觉问题？如何评估 RAG 系统的效果？

---

##### 🔹 方向B：智能体

**核心内容**
- Function Calling / Tool Use
- ReAct 框架
- Planning（计划能力）
- Agent 架构设计（单Agent vs 多Agent）

**任务**
- 实战：实现一个能调用工具（如计算器、搜索引擎）的 **ReAct Agent**
- 讨论：Agent 的规划与决策机制的局限性，如何提升鲁棒性？

---

## 三、深度保障建议

### 1. 会前准备要求

| 要求             | 具体内容                                                     |
| ---------------- | ------------------------------------------------------------ |
| **最低预习门槛** | • 所有人必须读完当周**核心论文的摘要、引言和结论**<br/>• 主讲人提前一天（周二）发送讲义草稿或核心问题列表 |
| **代码准备**     | • 连接者准备好**最小可复现示例**（Colab Notebook 或本地脚本），确保能在5分钟内跑通核心概念 |
| **问题清单**     | • 挑战者提前整理 **3-5 个核心问题**，并在会前（周三下午）分享到群里，让大家带着思考参会 |

---

### 2. 会中形式要求

| 形式         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **白板推导** | 强制主讲人手写公式、画流程图。只有写出来才算真懂。           |
| **代码对照** | 理论讲解后立即看代码实现，建立直观理解。**“讲完公式，看代码里这行在哪？”** |
| **禁止照读** | 讲解时不能照着论文/PPT念。如果内容复杂，念完后必须用自己的语言复述，并辅以例子讲透彻。 |

---

### 3. 学习产出沉淀

建议共同维护一个 Git 知识库：
```
    Architect-of-the-Tower-of-Babel/
    ├── week01-word2vec/
    │   ├── notes.md          # 主讲人笔记 + 讨论精华
    │   ├── questions.md      # 挑战者问题及全员讨论答案
    │   ├── demo.ipynb        # 连接者代码演示
    │   └── papers/           # 本周核心论文PDF
    ├── week02-rnn/
    │   └── ...
    ├── resources/            # 长期维护的资料索引
    │   ├── awesome-papers.md
    │   ├── useful-blogs.md
    │   └── cheatsheets/      # 公式速查表
    └── README.md             # 小组介绍与规划
```
---

### 4. 滚动回顾机制

- **每月一次**：在当月最后一次会议的最后15分钟，快速回顾本月学过的所有核心概念，讨论它们之间的联系。例如：
  - 第4周：Attention如何解决了第2周RNN的长距离依赖问题？
  - 第7周：GPT的自回归方式和第1周的统计语言模型有何渊源？



## 四、学习资源推荐

### 📚 经典教材

| 教材                             | 作者                           | 说明                 |
| -------------------------------- | ------------------------------ | -------------------- |
| *Speech and Language Processing* | Dan Jurafsky & James H. Martin | NLP 圣经级教材       |
| 《神经网络与深度学习》           | 邱锡鹏                         | 国内优秀深度学习教材 |
| *Natural Language Processing*    | Jacob Eisenstein               | 理论扎实，适合深入   |

---

### 🔗 优质博客与代码

| 资源                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) | Transformer 可视化讲解的经典之作                             |
| [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) | 逐行代码解析，手把手教你实现Transformer                      |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers/) | 主流预训练模型库，实战必备                                   |
| [Jay Alammar's Blog](http://jalammar.github.io/)             | 可视化教程系列，必读                                         |
| **LLM Course**                                               | https://github.com/mlabonne/llm-course 一个非常棒的LLM学习路线图，包含理论和代码。 |
| **Zotero**                                                   | https://www.zotero.org/ 强烈建议使用Zotero建立团队共享文献库，方便管理所有论文。 |

---

### 📄 论文管理

- 建议使用 **Zotero** 建立团队共享文献库
- 主要论文来源：ACL, EMNLP, NAACL, ICLR, NeurIPS

---

## 🎯 结语

三人小组的优势在于：**灵活、强互动、深度参与**。

只要坚持：
- ✅ **人人都要讲**
- ✅ **人人都要问**
- ✅ **人人都要写代码**
- ✅ **人人都要回头看**（定期回顾）

12周后，你们对 NLP 的理解将有质的飞跃！

---


**祝学习顺利！** 🚀



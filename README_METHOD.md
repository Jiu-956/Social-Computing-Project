# 方法实现与研究叙事

这份文档说明当前项目的方法是如何组织的，以及为什么这套比较框架适合讲出一个清晰的研究故事。

## 1. 研究主线

当前项目围绕一个很明确的问题展开：

**单个账号像不像机器人，和一群账号是不是在协同行动，其实是两个层次的问题。**

因此项目把整条流水线拆成两层：

1. 账号级检测：判断每个账号是 bot 还是 human
2. 群体级分析：判断高风险账号之间是否形成结构化社群

这也是为什么项目同时保留三类方法：

- 基于特征的方法
- 基于文本的方法
- 基于图的方法

并进一步扩展到：

- 图嵌入方法：DeepWalk、Node2Vec
- 图神经网络方法：GCN、BotRGCN
- 社区发现与模块度分析

## 2. 为什么要保留 baseline

原有的机器学习方法不是“旧代码”，而是故事里的第一层证据。

它们回答的是：

- 只看账号画像和简单文本，能做到什么程度？
- 把图结构特征和随机游走嵌入加进来，能提升多少？

因此当前 baseline 仍然保留：

- `profile_text_logreg`
- `graph_profile_rf`
- `full_logreg`

这三组模型提供一个稳定的对照系，后面所有增强方法都可以用它们来比较增益。

## 3. 数据准备层是怎么实现的

实现文件：`code/data.py`

这一层做的事情是：

- 从 `label.csv` 和 `split.csv` 得到有标签用户
- 从 `edge.csv` 提取 `follow / friend / post` 等关系
- 从 `node.json` 流式解析用户属性和 tweet 文本
- 为每个用户构造画像字段、图统计字段和合并文本字段

输出会进入 `result/cache/`，供后续所有实验复用。

这里的核心设计是：

- **分类只在有标签用户上评估**
- **图上下文尽量保留**
- **tweet 文本不直接作为节点做图学习，而是先转成用户级文本表示**

## 4. 图表示：DeepWalk 和 Node2Vec

实现文件：`code/features.py`

当前项目先构建用户关系图，然后同时训练两种随机游走嵌入：

- DeepWalk：无偏随机游走
- Node2Vec：带 `p/q` 偏置的随机游走

为什么两者都保留：

- DeepWalk 是经典图嵌入 baseline
- Node2Vec 可以更灵活地在“局部社区结构”和“更远的结构角色”之间取平衡

在当前代码里：

- DeepWalk 产物写到 `deepwalk_embeddings.joblib`
- Node2Vec 产物写到 `node2vec_embeddings.joblib`
- 合并后的图嵌入写到 `graph_embeddings.joblib`

这让后续实验可以显式比较：

- 传统图统计特征是否够用
- DeepWalk 能带来多少提升
- Node2Vec 是否比 DeepWalk 更适合机器人群体结构

## 5. 文本表示：从 TF-IDF 到 Transformer

实现文件：

- `code/experiments.py`
- `code/text_embeddings.py`

文本部分分成两层：

### 5.1 原始 baseline 文本表示

项目原有文本表示是：

- 用户简介 + 采样 tweet 合并成 `combined_text`
- 用 `TfidfVectorizer` 提取稀疏文本特征

这对应的是一个非常典型、可复现、可解释的 baseline。

### 5.2 更强文本编码器

现在新增了 Transformer 文本编码路径：

- 使用 `transformers` 的 `AutoTokenizer + AutoModel`
- 对文本做 mean pooling
- 输出稠密语义向量
- 缓存到 `result/cache/text_embeddings_*.joblib`

它的意义不是取代 TF-IDF，而是把故事推进一步：

- TF-IDF 更像关键词匹配
- Transformer 更像语义表示

如果 Transformer 模型有效，通常意味着模型不只是抓住了“spam 词”，而是学到了更稳定的语义模式。

## 6. 分类实验如何组织

实现文件：`code/experiments.py`

当前实验被组织成统一的对比框架，主要包含这些层次：

### 6.1 原始 baseline

- `profile_text_logreg`
- `graph_profile_rf`
- `full_logreg`

### 6.2 图嵌入增强

- `graph_node2vec_rf`
- `full_node2vec_logreg`

这组实验回答：

**把 DeepWalk 换成 Node2Vec 后，结构表示是否更好？**

### 6.3 强文本编码增强

- `transformer_profile_logreg`
- `transformer_graph_logreg`

这组实验回答：

**更强的文本语义表示，能否补足传统 TF-IDF 的局限？**

### 6.4 图神经网络

- `gcn_transformer`
- `botrgcn_transformer`

这组实验回答：

**仅靠“先提特征再分类”是否已经足够，还是需要让节点特征在图上传播？**

## 7. GCN 和 BotRGCN 是怎么落地的

实现文件：`code/gnn.py`

### 7.1 GCN

GCN 的核心思路是：

- 节点先有自己的特征
- 再通过图邻接矩阵把邻居信息聚合进来
- 最后在图上传播后的表示上分类

在当前实现里，GCN 使用：

- 两层图卷积
- 训练集监督
- 验证集 F1 做 early stopping

### 7.2 BotRGCN

BotRGCN 的关键不是“普通图卷积”，而是**关系感知**。

当前项目把边关系拆成不同 relation，例如：

- `follow`
- `friend`

然后分别建立 relation-specific adjacency，再在每种关系上做独立线性变换并聚合。

这样做的研究意义是：

- 普通 GCN 假设所有边类型作用相同
- BotRGCN 假设“关注关系”和“好友关系”的传播意义不一样

如果 BotRGCN 比 GCN 更强，就可以讲出一个更完整的故事：

**机器人协同行为不仅体现在有没有连接，更体现在是什么类型的连接。**

## 8. 群体发现：为什么先分类，再做结构分析

实现文件：

- `code/experiments.py`
- `code/community.py`

项目不是直接对全图做社区分析，而是采用两阶段流程：

1. 先用最佳分类器找出高风险 bot 候选
2. 再在候选子图上做聚类与社区发现

这样做的原因很现实：

- 直接对全图做社区发现，容易把大量正常用户的结构混进去
- 先筛出高风险候选，再分析其内部结构，更符合“协同行为检测”的目标

## 9. 两种群体分析视角

### 9.1 向量空间聚类

`cluster` 命令做的是：

- 在候选 bot 上取画像 + 图特征 + 图嵌入
- 标准化
- PCA 降维
- DBSCAN 或 Spectral Clustering

它回答的是：

**这些高风险账号在表示空间里能不能自动聚成团？**

### 9.2 社区发现与模块度分析

`community` 命令做的是：

- 在候选 bot 构成的子图上
- 跑 Louvain 和 greedy modularity
- 输出社区划分、社区规模和 modularity

它回答的是：

**这些高风险账号在真实社交关系图中，是否形成了非随机的社群结构？**

这两者的意义不同：

- 聚类：更偏表示空间
- 社区发现：更偏真实网络结构

如果两者同时支持“这些 bot 形成了紧密群体”，故事就会非常完整。

## 10. 自动报告在故事里起什么作用

实现文件：`code/reporting.py`

自动报告不是简单把 CSV 拼起来，而是服务于研究叙事。

报告会自动汇总：

- 最优模型是谁
- 相对 strongest baseline 的提升是多少
- 各 family 的最好结果
- 聚类得到多少可疑群体
- 哪种社区发现方法的模块度最高

最终输出 `result/report.md`，它能直接支持汇报或论文写作中的“实验结果与分析”部分。

## 11. 当前最适合讲的故事

基于这套框架，一个比较完整、自然的故事可以这样讲：

1. **第一层：传统可解释 baseline**
   先证明账号画像、简单文本和图统计特征本身就能识别一部分 bot。

2. **第二层：更强表示提升检测性能**
   再证明 Node2Vec 和 Transformer 文本编码器能比原始表示更好地捕获结构与语义信号。

3. **第三层：图神经网络进一步利用协同结构**
   GCN 和 BotRGCN 不只是“看单个账号”，而是在图上传播信息，因而更适合处理协同行为。

4. **第四层：群体结构证据闭环**
   即便账号级分类很强，也还需要说明这些高风险 bot 是否真的形成群体。
   聚类、社区发现和模块度分析正好提供了这个结构层面的证据。

因此最终故事不是“哪个模型分数更高”，而是：

**从账号属性、文本语义、图结构到社区组织形式，多层证据共同支持机器人协同行为的存在。**

## 12. 一句话总结

当前项目已经从“传统机器学习 bot 检测 baseline”升级成了一套多方法、可比较、可讲故事的社会计算实验框架：既能比较单账号检测性能，也能分析高风险账号之间是否形成结构化协同群体。

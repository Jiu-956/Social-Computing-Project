# 基于 TwiBot-20 的社交机器人检测实验报告

## 1. 研究目标
本项目围绕 TwiBot-20 数据集完成 bot / human 二分类实验，并严格按照论文常见的六类方法组织实验：基于特征、基于文本、基于图、基于特征和文本、基于特征和图、基于特征、文本和图。

## 2. 数据与处理
- 图用户总数：229580
- 有标签用户数：11826
- support 节点数：217754
- 用户关系边数：227979
- 采样 tweet 数：2343696

数据准备阶段对 `node.json` 采用流式解析，避免 5.7GB 原始文件一次性载入内存；同时保留 support 节点，为 GCN、GAT、BotRGCN 风格模型提供完整的邻居上下文。

## 3. 六类方法设计
- 基于特征：只使用用户画像和账号属性特征。
- 基于文本：只使用简介文本和 tweet 文本表示。
- 基于图：只使用图统计特征或图嵌入。
- 基于特征和文本：将结构化账号特征与文本特征拼接后分类。
- 基于特征和图：将账号属性与图统计或 Node2Vec 图嵌入融合。
- 基于特征、文本和图：同时输入 description 表示、tweet 表示、数值属性、类别属性与图结构，最接近论文中的 FTG 思路。

## 4. 验证集表现
| experiment | family_cn | accuracy | precision | recall | f1 | auc_roc |
| --- | --- | --- | --- | --- | --- | --- |
| feature_text_tfidf_logistic_regression | 基于特征和文本 | 0.8385 | 0.7946 | 0.9532 | 0.8667 | 0.9166 |
| feature_text_graph_botrgcn | 基于特征、文本和图 | 0.8326 | 0.7747 | 0.9816 | 0.8659 | 0.8815 |
| feature_text_graph_tfidf_node2vec_logistic_regression | 基于特征、文本和图 | 0.8393 | 0.8014 | 0.9417 | 0.8659 | 0.9113 |
| feature_text_transformer_logistic_regression | 基于特征和文本 | 0.8452 | 0.8368 | 0.8933 | 0.8641 | 0.9146 |
| feature_text_graph_gat | 基于特征、文本和图 | 0.8296 | 0.7788 | 0.9647 | 0.8618 | 0.8909 |
| feature_only_random_forest | 基于特征 | 0.8047 | 0.7501 | 0.9678 | 0.8452 | 0.8456 |
| feature_only_logistic_regression | 基于特征 | 0.7962 | 0.7326 | 0.9923 | 0.8429 | 0.8407 |
| feature_graph_random_forest | 基于特征和图 | 0.8008 | 0.7482 | 0.9624 | 0.8419 | 0.8436 |
| feature_graph_node2vec_logistic_regression | 基于特征和图 | 0.8051 | 0.7621 | 0.9394 | 0.8415 | 0.8579 |
| feature_text_graph_gcn | 基于特征、文本和图 | 0.8165 | 0.8084 | 0.8741 | 0.8400 | 0.8766 |
| text_only_tfidf_logistic_regression | 基于文本 | 0.7112 | 0.7303 | 0.7544 | 0.7422 | 0.7752 |
| text_only_transformer_logistic_regression | 基于文本 | 0.6786 | 0.7195 | 0.6830 | 0.7008 | 0.7476 |

## 5. 测试集表现
| experiment | family_cn | accuracy | precision | recall | f1 | auc_roc |
| --- | --- | --- | --- | --- | --- | --- |
| feature_text_graph_botrgcn | 基于特征、文本和图 | 0.8436 | 0.7883 | 0.9719 | 0.8705 | 0.8808 |
| feature_text_graph_gat | 基于特征、文本和图 | 0.8385 | 0.7904 | 0.9547 | 0.8648 | 0.9033 |
| feature_text_tfidf_logistic_regression | 基于特征和文本 | 0.8402 | 0.7987 | 0.9422 | 0.8645 | 0.9167 |
| feature_only_logistic_regression | 基于特征 | 0.8174 | 0.7506 | 0.9922 | 0.8546 | 0.8583 |
| feature_text_graph_tfidf_node2vec_logistic_regression | 基于特征、文本和图 | 0.8301 | 0.7995 | 0.9156 | 0.8536 | 0.9084 |
| feature_text_transformer_logistic_regression | 基于特征和文本 | 0.8352 | 0.8277 | 0.8781 | 0.8522 | 0.9147 |
| feature_graph_random_forest | 基于特征和图 | 0.8132 | 0.7609 | 0.9547 | 0.8468 | 0.8640 |
| feature_only_random_forest | 基于特征 | 0.8090 | 0.7607 | 0.9437 | 0.8424 | 0.8649 |
| feature_graph_node2vec_logistic_regression | 基于特征和图 | 0.8090 | 0.7717 | 0.9187 | 0.8388 | 0.8638 |
| feature_text_graph_gcn | 基于特征、文本和图 | 0.7971 | 0.8012 | 0.8313 | 0.8160 | 0.8762 |
| text_only_tfidf_logistic_regression | 基于文本 | 0.6974 | 0.7104 | 0.7438 | 0.7267 | 0.7609 |
| graph_only_structure_random_forest | 基于图 | 0.6526 | 0.6944 | 0.6391 | 0.6656 | 0.7042 |

## 6. 按方法族汇总
| family_cn | split | best_experiment | accuracy | f1 | auc_roc |
| --- | --- | --- | --- | --- | --- |
| 基于特征、文本和图 | test | feature_text_graph_botrgcn | 0.8436 | 0.8705 | 0.8808 |
| 基于特征和文本 | test | feature_text_tfidf_logistic_regression | 0.8402 | 0.8645 | 0.9167 |
| 基于特征 | test | feature_only_logistic_regression | 0.8174 | 0.8546 | 0.8583 |
| 基于特征和图 | test | feature_graph_random_forest | 0.8132 | 0.8468 | 0.8640 |
| 基于文本 | test | text_only_tfidf_logistic_regression | 0.6974 | 0.7267 | 0.7609 |
| 基于图 | test | graph_only_structure_random_forest | 0.6526 | 0.6656 | 0.7042 |
| 基于特征和文本 | val | feature_text_tfidf_logistic_regression | 0.8385 | 0.8667 | 0.9166 |
| 基于特征、文本和图 | val | feature_text_graph_botrgcn | 0.8326 | 0.8659 | 0.8815 |
| 基于特征 | val | feature_only_random_forest | 0.8047 | 0.8452 | 0.8456 |
| 基于特征和图 | val | feature_graph_random_forest | 0.8008 | 0.8419 | 0.8436 |
| 基于文本 | val | text_only_tfidf_logistic_regression | 0.7112 | 0.7422 | 0.7752 |
| 基于图 | val | graph_only_structure_random_forest | 0.6486 | 0.6701 | 0.7047 |

## 7. 结果解读
- 按验证集 F1 选择的最佳模型是 `feature_text_tfidf_logistic_regression`，验证集 F1 为 0.8667。
- 测试集按准确率最高的模型是 `feature_text_graph_botrgcn`，Accuracy 为 0.8436，F1 为 0.8705。
- 测试集按 F1 最高的模型是 `feature_text_graph_botrgcn`，F1 为 0.8705，Accuracy 为 0.8436。
- 本次最强图神经网络模型是 `feature_text_graph_botrgcn`，测试集 Accuracy 为 0.8436，F1 为 0.8705。
- 相比旧版简化图模型，当前 FTG 图模型把 description、tweet、数值属性、类别属性和关系边同时输入图网络，更接近论文和官方仓库里的实现组织方式。
- 如果只看 Accuracy，图方法已经进入全项目第一梯队；如果看 F1，基于特征和文本的方法仍然略占优势，这说明当前任务里文本语义和用户属性依然非常重要。
- BotRGCN 风格模型优于 GCN，说明区分 follow 与 friend 关系类型是有价值的，这与论文中“关系感知传播更有效”的结论一致。

## 8. 与论文和官方仓库的关系
本项目参考了论文 2206.04564v6 以及官方仓库 `LuoUndergradXJTU/TwiBot-22` 中对 F / T / G / FT / FG / FTG 六类方法的组织方式。
图模型部分重点借鉴了官方仓库在 TwiBot-20 上的输入结构：description 表示、tweet 表示、数值属性、类别属性和关系边共同进入 GCN / GAT / BotRGCN 风格模型。
需要说明的是，本项目仍然属于课程项目级的复现与重实现，而不是逐文件复刻官方仓库，因此结果可以用于趋势比较，但不应直接表述为严格复现论文原始成绩。

## 9. 可视化与附件
- `artifacts/figures/model_comparison.png`：测试集模型 F1 对比图
- `artifacts/figures/node2vec_projection.png`：Node2Vec 二维投影
- `artifacts/figures/feature_importance.png`：传统特征重要性
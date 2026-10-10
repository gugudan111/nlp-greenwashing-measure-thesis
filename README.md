# 基于 NLP 的企业 ESG 报告"漂绿"行为文本测度构建（毕业论文预演模拟稿）

> 本课程作业依《管理研究中的 NLP 测度构建方法》所授 **NLP 测度五步主线**（语料获取 → 文本预处理 → 特征提取 → 指标构建 → 信效度检验）完成，目标是走通"把一段文字变成一个能进论文的变量"的完整流程。
>
> ⚠️ **课程预演模拟稿**：实验语料为按真实 ESG 披露语言规律**合成的可复现数据**（附随机种子），用于演示方法论；测度方法可直接迁移至真实 ESG 报告文本。

## 研究问题

**RQ**：如何基于自然语言处理技术，构建客观、可复现、可量化企业 ESG 报告中"漂绿"程度的文本测度 GWI，并验证其信度与效度？

沿 Concept → Construct → Indicator → Measure → Variable 的翻译链，将"漂绿"操作化为"承诺性—模糊性—证据稀缺性"三维语言信号。

## 方法：三路径融合的 GWI 测度

| 步骤 | 内容 | 实现 |
|---|---|---|
| ① 语料获取 | 合成 ESG 披露语料（含模糊边界句） | `code/generate_synthetic_data.py` |
| ② 文本预处理 | jieba 精确分词 + 停用词（含财经领域词） | `build_measure.py` |
| ③ 特征提取 | 词典法 / 词嵌入（PPMI+SVD）/ 有监督学习（多项式 NB） | `build_measure.py` |
| ④ 指标构建 | L1 计数 → L2 标准化比率 → L3 等权融合 | `build_measure.py` |
| ⑤ 信效度检验 | 内容 / 聚合 / 区分 / 预测 四类效度 | `build_measure.py` |

- **GWI_dict**：`ln(1+承诺+模糊) / ln(1+证据+1)`
- **GWI_ml**：朴素贝叶斯后验概率（纯 NumPy，5 折交叉验证）
- **GWI_emb**：句子向量与"漂绿修辞"语义锚点的余弦相似度（语料 PPMI → 截断 SVD 得到词向量）
- **GWI**：三分量 min-max 标准化后等权融合

## 仓库结构

```
paper/论文.md                  预演论文全文（≥6000 字，标准学术结构）
code/generate_synthetic_data.py 合成语料生成（固定随机种子）
code/build_measure.py           测度构建 + 融合 + 全部图表
data/esg_sentences.csv          句子级语料 + 金标准标签（测度结果表）
data/esg_firm_meta.csv          企业级代理变量（环境处罚 / 第三方评分）
figures/                       六张实验结果图表
references.bib                 参考文献 BibTeX 源
results_summary.txt            数值结果摘要
```

## 运行方式（无需 GPU）

```bash
pip install -r requirements.txt
python code/generate_synthetic_data.py
python code/build_measure.py
```

环境：`Python 3.13` + `numpy` + `matplotlib` + `jieba`。

## 实验结果速览

（固定随机种子，结果可完全复现）

- 样本：句子 ~1320 条，企业 120 家，行业 8 个
- 监督分量（5 折 CV）：句子级 **AUC≈0.99**，AP≈0.99
- 三分量收敛（Pearson r）：均 > 0.95（聚合效度）
- 外部效度：GWI 与"环境处罚强度" r=0.64、与"第三方环境评分" r=−0.71（均 p<0.01）
- 预测效度回归：Penalty = 0.38 + 3.89·GWI（t=8.99, R²=0.41）
- 区分效度：高 GWI 组证据密度 3.83 vs 低分组 6.57（Welch t≈−13.8, p<0.001）

## 图表索引

| 文件 | 内容 |
|---|---|
| `figures/correlation_heatmap.png` | 三分量 + 综合 GWI 相关性热力图（聚合效度） |
| `figures/gwi_distribution.png`   | 企业层 GWI 分布直方图 |
| `figures/gwi_by_industry.png`    | 分行业平均 GWI 柱状图 |
| `figures/roc_curve.png`          | 监督分量 ROC 曲线 |
| `figures/pr_curve.png`           | 监督分量 PR 曲线 |
| `figures/gwi_vs_penalty.png`     | GWI 与环境处罚散点（外部效度） |

## 说明与许可证

- 本仓库为**课程作业预演**，参考文献用于方法演示与格式示范，正式提交前请通过学校图书馆数据库逐条核对原文信息。
- 代码以 MIT 许可证开源；论文正文以 CC BY 4.0 授权，转载请注明来源。
- 真实 ESG 报告文本请通过巨潮资讯网、CNRDS、Wind 等合规渠道获取，并注意时间对齐与样本筛选。

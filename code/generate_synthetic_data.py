# -*- coding: utf-8 -*-
"""
generate_synthetic_data.py
--------------------------
生成可复现的合成 ESG 披露语料，用于演示"漂绿"文本测度的构建。

数据按真实 ESG 环境章节的常见句式构造，包含三类句子：
  - 高漂绿（重承诺/重模糊/轻证据）
  - 低漂绿/实质披露（含具体量化行动）
  - 模糊边界句（承诺性语言 + 量化证据并存，标签随机）——模拟真实披露中
    "承诺与证据并存"导致人工标注存在不确定性的情形，使分类器无法达到完美分离，
    从而得到贴近实证发现的 AUC。

语料覆盖若干虚拟行业与企业，并附企业级代理变量（环境处罚强度、第三方环境评分），
由潜在漂绿倾向驱动 + 独立噪声生成，用于外部效度检验。

运行：python code/generate_synthetic_data.py
依赖：标准库（无需第三方包）
说明：固定随机种子，保证结果可复现。此数据为课程方法演示之用，非真实企业数据。
"""
import csv
import os
import random

SEED = 20240410
random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_SENT = os.path.join(DATA_DIR, "esg_sentences.csv")
OUT_FIRM = os.path.join(DATA_DIR, "esg_firm_meta.csv")

# 行业 -> 高排放倾向锚点（仅用于让潜在漂绿倾向 g 在行业间有差异，增强外部效度演示）
INDUSTRIES = {
    "化工": 0.90, "电力": 0.90, "钢铁": 0.88, "建材": 0.82,
    "医药": 0.38, "消费": 0.42, "金融": 0.20, "信息技术": 0.25,
}
FIRMS_PER_INDUSTRY = 15
AMBIGUOUS_RATE = 0.12  # 模糊边界句占比

COMMIT = ["致力于", "力争", "计划", "将", "承诺", "目标", "推进", "打造", "树立",
          "引领", "全面提升", "牢固树立", "矢志", "构建", "融入", "完善", "增强"]
VAGUE = ["持续", "逐步", "有望", "进一步", "不断", "显著", "积极", "努力", "全面",
         "更大", "突破", "始终", "深入", "稳步", "优化", "迈向"]

HIGH_TEMPLATES = [
    "公司{commit}打造绿色低碳标杆，{vague}引领行业转型。",
    "我们{commit}{vague}提升环境管理水平，有望实现更大突破。",
    "公司{commit}推进可持续发展，{vague}优化能源结构。",
    "本企业{commit}树立生态文明理念，{vague}履行环保责任。",
    "我们将{commit}构建绿色制造体系，{vague}增强社会责任。",
    "公司{commit}全面提升环保绩效，{vague}迈向碳中和目标。",
    "管理层{commit}把绿色发展融入战略，{vague}完善环境治理。",
]

LOW_TEMPLATES = [
    "公司已完成{n}兆瓦光伏发电项目建设，年减排{m}万吨二氧化碳。",
    "本报告期累计投入{x}亿元用于脱硫脱硝改造，减排二氧化硫{y}万吨。",
    "已建成{n}兆瓦余热发电装置，节约标准煤{m}万吨。",
    "通过{act}实现碳强度下降{p}%，累计植树{q}万棵。",
    "投入{x}亿元建成循环经济产业园，资源循环利用率达{r}%。",
    "已建成{n}兆瓦风电项目，年节约标煤{m}万吨，减排二氧化碳{y}万吨。",
    "累计投入{x}亿元完成{act}，单位产品能耗下降{p}%。",
]

AMBIG_TEMPLATES = [
    "公司{commit}{vague}推进绿色转型，报告期内减排{n}万吨二氧化碳。",
    "我们{commit}提升环境绩效，已建成{n}兆瓦光伏项目。",
    "公司{commit}打造绿色标杆，投入{x}亿元用于节能改造。",
]


def rnum(a, b, dec=1):
    return f"{round(random.uniform(a, b), dec)}"


def gen_sentence(kind):
    if kind == "high":
        t = random.choice(HIGH_TEMPLATES)
        return t.replace("{commit}", random.choice(COMMIT)).replace("{vague}", random.choice(VAGUE))
    if kind == "low":
        t = random.choice(LOW_TEMPLATES)
        return (t.replace("{n}", rnum(20, 200, 0))
                 .replace("{m}", rnum(3, 30))
                 .replace("{x}", rnum(0.5, 8))
                 .replace("{y}", rnum(2, 15))
                 .replace("{p}", rnum(5, 30, 0))
                 .replace("{q}", rnum(5, 30, 0))
                 .replace("{r}", rnum(40, 95, 0))
                 .replace("{act}", random.choice(["余热回收", "能效改造", "绿电采购", "污水回用"])))
    # ambiguous：承诺/模糊语言 + 量化证据并存，标签随机（边界情形）
    t = random.choice(AMBIG_TEMPLATES)
    return (t.replace("{commit}", random.choice(COMMIT))
             .replace("{vague}", random.choice(VAGUE))
             .replace("{n}", rnum(20, 200, 0))
             .replace("{x}", rnum(0.5, 8)))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    sent_rows = []
    firm_rows = []
    firm_id = 0
    for ind, bias in INDUSTRIES.items():
        for _ in range(FIRMS_PER_INDUSTRY):
            firm_id += 1
            fid = f"F{firm_id:03d}"
            # 潜在漂绿倾向 g（0~1）：行业锚点 + 个体异质 + 边界裁剪
            g = random.betavariate(2, 5) + (bias - 0.5) * 0.35
            g = min(0.95, max(0.05, g))
            n_sent = random.randint(8, 14)
            for i in range(n_sent):
                if random.random() < AMBIGUOUS_RATE:
                    kind = "amb"
                    label = random.randint(0, 1)  # 边界句标签随机
                else:
                    high = random.random() < g
                    kind = "high" if high else "low"
                    label = int(high)
                sent = gen_sentence(kind)
                sent_rows.append((fid, ind, i, sent, label))
            # 企业级代理变量：由潜在 g 驱动 + 独立噪声（模拟真实世界可观测结果）
            penalty = max(0.0, round(g * random.uniform(3.0, 8.0) + random.gauss(0, 0.8), 2))
            env_score = round(max(0.0, 100 - g * 70 + random.gauss(0, 8)), 1)
            firm_rows.append((fid, ind, round(g, 3), penalty, env_score))

    with open(OUT_SENT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "industry", "sent_idx", "sentence", "label"])
        w.writerows(sent_rows)
    with open(OUT_FIRM, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "industry", "latent_g", "penalty_intensity", "env_score"])
        w.writerows(firm_rows)

    n_amb = sum(1 for r in sent_rows if r[4] == 1 and "减排" in r[3] and any(c.isdigit() for c in r[3]) and "建成" not in r[3] and "已完成" not in r[3] and "投入" not in r[3] and "累计" not in r[3])  # 仅粗略计数
    n_high = sum(1 for r in sent_rows if r[4] == 1)
    print(f"[OK] 生成句子 {len(sent_rows)} 条（高漂绿标签 {n_high} / 低漂绿标签 {len(sent_rows)-n_high}）")
    print(f"[OK] 生成企业 {len(firm_rows)} 家，覆盖行业 {len(INDUSTRIES)} 个（含约 {int(AMBIGUOUS_RATE*100)}% 模糊边界句）")
    print(f"[OK] 写入: {OUT_SENT}")
    print(f"[OK] 写入: {OUT_FIRM}")


if __name__ == "__main__":
    main()

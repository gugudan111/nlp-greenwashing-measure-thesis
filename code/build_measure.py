# -*- coding: utf-8 -*-
"""
build_measure.py
----------------
基于合成 ESG 披露语料，演示"漂绿"文本测度（GWI）的三路径构建与融合：
  1) 词典法          GWI_dict
  2) 有监督机器学习   GWI_ml   （多项式朴素贝叶斯，纯 NumPy 实现，5 折交叉验证）
  3) 词向量嵌入相似度 GWI_emb  （语料共现 -> PPMI -> 截断 SVD 得到词向量，语义锚点余弦相似度）
综合测度 GWI = 三分量 min-max 标准化后等权融合。
脚本同时产出全部实验图表（见 figures/）与结果摘要（results_summary.txt）。

运行：python code/build_measure.py
依赖：numpy, matplotlib, jieba
"""
import os
import re
import csv
import math
import numpy as np
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jieba.setLogLevel(20)
np.random.seed(20240410)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
SENT_CSV = os.path.join(DATA_DIR, "esg_sentences.csv")
FIRM_CSV = os.path.join(DATA_DIR, "esg_firm_meta.csv")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 词典（与数据生成模板对齐）----------
COMMIT_WORDS = {"致力于", "力争", "计划", "将", "承诺", "目标", "推进", "打造", "树立",
                "引领", "全面提升", "牢固树立", "矢志", "构建", "融入", "完善", "增强"}
VAGUE_WORDS = {"持续", "逐步", "有望", "进一步", "不断", "显著", "积极", "努力", "全面",
               "更大", "突破", "始终", "深入", "稳步", "优化", "迈向"}
EVIDENCE_WORDS = {"减排", "建成", "投入", "节约", "发电", "光伏", "余热", "脱硫", "脱硝",
                  "循环", "植树", "碳强度", "标准煤", "二氧化碳", "万元", "亿元", "万吨",
                  "兆瓦", "千瓦", "已完成", "已建成", "已实现", "累计", "降低", "减少",
                  "能效", "污水", "回用", "绿电", "采购"}
ANCHOR_WORDS = list(COMMIT_WORDS | VAGUE_WORDS)  # 嵌入语义锚点（漂绿修辞词）

STOPWORDS = {"的", "了", "在", "是", "和", "与", "及", "等", "我们", "公司", "企业",
             "本报告期", "通过", "已", "本", "该", "其", "为", "对", "把", "向", "由",
             "以", "并", "也", "都", "就", "而", "但", "则", "之", "中", "上", "下",
             "个", "这", "那", "年", "月", "日", "至", "从", "把", "被", "让", "给"}
NUM_RE = re.compile(r"\d+\.?\d*")


# ---------- 工具函数 ----------
def tokenize(text):
    toks = [t for t in jieba.lcut(text) if t.strip() and t not in STOPWORDS]
    return toks


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[j + 1] == x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def auc(y_true, y_score):
    y_true = np.asarray(y_true, int); y_score = np.asarray(y_score, float)
    order = np.argsort(y_score, kind="mergesort")
    ys = y_score[order]; yt = y_true[order]
    ranks = np.empty(len(ys))
    i = 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[i:j + 1] = avg
        i = j + 1
    n_pos = int(yt.sum()); n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s = ranks[yt == 1].sum()
    return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_pr(y_true, y_score):
    y_true = np.asarray(y_true, int); y_score = np.asarray(y_score, float)
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    P = y.sum(); N = len(y) - P
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.concatenate([[0], np.cumsum(1 - y)])
    tpr = tp / P
    fpr = fp / N
    rec = tp / P
    prec = np.where(tp + fp > 0, tp / (tp + fp + 1e-12), 1.0)
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    return fpr, tpr, prec, rec, ap


def welch_t(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    t = (mx - my) / math.sqrt(vx / nx + vy / ny)
    return t, mx, my


def ols1(x, y):
    """单变量 OLS: y = b0 + b1*x。返回 (b0, b1, se_b1, t_b1, r2)。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    sxy = ((x - xb) * (y - yb)).sum()
    b1 = sxy / sxx
    b0 = yb - b1 * xb
    yhat = b0 + b1 * x
    rss = ((y - yhat) ** 2).sum()
    sigma2 = rss / (n - 2)
    se_b1 = math.sqrt(sigma2 / sxx)
    t_b1 = b1 / se_b1
    r2 = 1 - rss / ((y - yb) ** 2).sum()
    return b0, b1, se_b1, t_b1, r2


# ---------- 朴素贝叶斯（多项式，Laplace 平滑）----------
def fit_nb(X, y):
    X = np.asarray(X); y = np.asarray(y, int)
    V = X.shape[1]
    out = {}
    for c in (0, 1):
        m = (y == c)
        Xc = X[m].astype(float)
        totals = Xc.sum(0)
        T = totals.sum()
        out[c] = (np.log(m.sum() / X.shape[0]),
                  np.log((totals + 1.0) / (T + V)))
    return out


def predict_p1(X, nb):
    X = np.asarray(X, float)
    s0 = X.dot(nb[0][1]) + nb[0][0]
    s1 = X.dot(nb[1][1]) + nb[1][0]
    m = np.maximum(s0, s1)
    e0 = np.exp(s0 - m); e1 = np.exp(s1 - m)
    return e1 / (e0 + e1)


# ---------- 主流程 ----------
def main():
    # 1) 读数据
    rows = []
    with open(SENT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["firm_id"], r["industry"], r["sentence"], int(r["label"])))
    firms = {}
    with open(FIRM_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            firms[r["firm_id"]] = (r["industry"], float(r["latent_g"]),
                                   float(r["penalty_intensity"]), float(r["env_score"]))

    sentences = [r[2] for r in rows]
    labels = np.array([r[3] for r in rows], int)
    firm_ids = [r[0] for r in rows]

    # 2) 分词 + 词典特征
    tok_list = [tokenize(s) for s in sentences]
    c_commit = np.array([sum(1 for t in toks if t in COMMIT_WORDS) for toks in tok_list], float)
    c_vague = np.array([sum(1 for t in toks if t in VAGUE_WORDS) for toks in tok_list], float)
    c_evid = np.array([sum(1 for t in toks if t in EVIDENCE_WORDS) +
                       len(NUM_RE.findall(sentences[i])) for i, toks in enumerate(tok_list)], float)
    gwi_dict_sent = np.log1p(c_commit + c_vague) / np.log1p(c_evid + 1.0)

    # 3) 有监督：BoW + 词典特征，5 折 CV
    from collections import Counter
    freq = Counter()
    for toks in tok_list:
        freq.update(set(toks))
    bow_vocab = [w for w, c in freq.items() if c >= 3][:800]
    bow_idx = {w: i for i, w in enumerate(bow_vocab)}
    Vb = len(bow_vocab)
    Xbow = np.zeros((len(rows), Vb + 3), float)
    for i, toks in enumerate(tok_list):
        for t in toks:
            if t in bow_idx:
                Xbow[i, bow_idx[t]] += 1.0
        Xbow[i, Vb] = c_commit[i]
        Xbow[i, Vb + 1] = c_vague[i]
        Xbow[i, Vb + 2] = c_evid[i]
    Xbow = Xbow.astype(int)

    idx0 = np.where(labels == 0)[0]; idx1 = np.where(labels == 1)[0]
    rng = np.random.RandomState(20240410)
    rng.shuffle(idx0); rng.shuffle(idx1)
    folds = 5
    oof = np.zeros(len(rows), float)
    for fi in range(folds):
        te0 = idx0[fi::folds]; te1 = idx1[fi::folds]
        te = np.concatenate([te0, te1])
        tr = np.ones(len(rows), bool); tr[te] = False
        nb = fit_nb(Xbow[tr], labels[tr])
        oof[te] = predict_p1(Xbow[te], nb)
    ml_auc = auc(labels, oof)
    fpr, tpr, prec, rec, ap = roc_pr(labels, oof)
    gwi_ml_sent = oof

    # 4) 嵌入：PPMI + 截断 SVD
    freq2 = Counter()
    for toks in tok_list:
        freq2.update(toks)
    emb_vocab = [w for w, c in freq2.items() if c >= 3][:500]
    eV = len(emb_vocab)
    eidx = {w: i for i, w in enumerate(emb_vocab)}
    cooc = np.zeros((eV, eV), float)
    freq_arr = np.zeros(eV, float)
    for toks in tok_list:
        present = sorted({eidx[t] for t in toks if t in eidx})
        for a in present:
            freq_arr[a] += 1
        for a in range(len(present)):
            for b in range(a, len(present)):
                cooc[present[a], present[b]] += 1
                if a != b:
                    cooc[present[b], present[a]] += 1
    Nsent = len(rows)
    ppmi = np.zeros((eV, eV), float)
    for a in range(eV):
        for b in range(a, eV):
            c = cooc[a, b]
            if c > 0 and freq_arr[a] > 0 and freq_arr[b] > 0:
                pmi = math.log(c * Nsent / (freq_arr[a] * freq_arr[b]))
                v = max(0.0, pmi)
            else:
                v = 0.0
            ppmi[a, b] = v
            ppmi[b, a] = v
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = 50
    W = U[:, :d] * S[:d]
    anchor = np.zeros(d, float)
    cnt = 0
    for w in ANCHOR_WORDS:
        if w in eidx:
            anchor += W[eidx[w]]
            cnt += 1
    if cnt > 0:
        anchor /= cnt
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
    sent_emb = np.zeros(len(rows), float)
    for i, toks in enumerate(tok_list):
        vs = [W[eidx[t]] for t in toks if t in eidx]
        if vs:
            v = np.mean(vs, 0)
            nv = np.linalg.norm(v)
            if nv > 0:
                sent_emb[i] = float(v.dot(anchor) / nv)
    gwi_emb_sent = sent_emb

    # 5) 企业级聚合 + 标准化融合
    uniq_firms = list(dict.fromkeys(firm_ids))
    F = len(uniq_firms)
    raw = np.zeros((F, 3), float)
    evid_density = np.zeros(F, float)
    n_sent_f = np.zeros(F, int)
    penalty = np.zeros(F, float)
    env_score = np.zeros(F, float)
    latent_g = np.zeros(F, float)
    industry = []
    fmap = {fid: k for k, fid in enumerate(uniq_firms)}
    for i, fid in enumerate(firm_ids):
        k = fmap[fid]
        raw[k, 0] += gwi_dict_sent[i]
        raw[k, 1] += gwi_ml_sent[i]
        raw[k, 2] += gwi_emb_sent[i]
        evid_density[k] += c_evid[i]
        n_sent_f[k] += 1
    for fid in uniq_firms:
        k = fmap[fid]
        ind, g, pen, es = firms[fid]
        industry.append(ind)
        penalty[k] = pen; env_score[k] = es; latent_g[k] = g
    raw /= n_sent_f[:, None]
    evid_density /= n_sent_f

    mins = raw.min(0); maxs = raw.max(0)
    norm = (raw - mins) / (maxs - mins + 1e-12)
    gwi = norm.mean(1)

    # 6) 效度检验
    corr_dict_pen = pearson(norm[:, 0], penalty)
    corr_ml_pen = pearson(norm[:, 1], penalty)
    corr_emb_pen = pearson(norm[:, 2], penalty)
    corr_gwi_pen = pearson(gwi, penalty)
    corr_gwi_env = pearson(gwi, env_score)
    corr_gwi_lat = pearson(gwi, latent_g)
    # 方法间收敛
    r_dm = pearson(norm[:, 0], norm[:, 1])
    r_de = pearson(norm[:, 0], norm[:, 2])
    r_me = pearson(norm[:, 1], norm[:, 2])

    # 判别效度：按 GWI 中位数分组
    med = np.median(gwi)
    hi = evid_density[gwi >= med]; lo = evid_density[gwi < med]
    t_stat, m_hi, m_lo = welch_t(hi, lo)

    # 6.5) 描述性统计与预测效度回归
    desc_vars = {"GWI_dict": norm[:, 0], "GWI_ml": norm[:, 1],
                 "GWI_emb": norm[:, 2], "GWI": gwi}
    desc_lines = ["=== 描述性统计（企业层，标准化后）==="]
    desc_stats = {}
    for nm, arr in desc_vars.items():
        arr = np.asarray(arr, float)
        st = (arr.mean(), arr.std(ddof=1), arr.min(), np.median(arr), arr.max())
        desc_stats[nm] = st
        desc_lines.append(f"  {nm}: mean={st[0]:.3f} sd={st[1]:.3f} "
                          f"min={st[2]:.3f} median={st[3]:.3f} max={st[4]:.3f}")
    b0p, b1p, se1p, t1p, r2p = ols1(gwi, penalty)
    b0e, b1e, se1e, t1e, r2e = ols1(gwi, env_score)
    reg_lines = ["=== 预测效度回归（OLS, 单变量）===",
                 f"  Penalty = {b0p:.3f} + {b1p:.3f}*GWI   (t={t1p:.2f}, R2={r2p:.3f})",
                 f"  EnvScore= {b0e:.3f} + {b1e:.3f}*GWI   (t={t1e:.2f}, R2={r2e:.3f})"]

    # 7) 绘图
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

    # 图1：相关性热力图（分量 + 综合）
    names = ["GWI_dict", "GWI_ml", "GWI_emb", "GWI"]
    M = np.corrcoef(np.column_stack([norm[:, 0], norm[:, 1], norm[:, 2], gwi]).T)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="YlOrRd", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="black" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title("Convergence of GWI Components (Pearson r)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png")); plt.close(fig)

    # 图2：GWI 分布
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(gwi, bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Composite GWI (firm-level)"); ax.set_ylabel("Number of firms")
    ax.set_title("Distribution of Corporate Greenwashing Index")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "gwi_distribution.png")); plt.close(fig)

    # 图3：分行业 GWI
    ind_order = ["化工", "电力", "钢铁", "建材", "医药", "消费", "金融", "信息技术"]
    eng = {"化工": "Chem", "电力": "Power", "钢铁": "Steel", "建材": "Cement",
           "医药": "Pharma", "消费": "Consumer", "金融": "Finance", "信息技术": "IT"}
    means = [gwi[[i for i, x in enumerate(industry) if x == ind]].mean() for ind in ind_order]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#C44E52" if m > np.median(means) else "#55A868" for m in means]
    ax.bar([eng[i] for i in ind_order], means, color=colors)
    ax.set_ylabel("Mean GWI"); ax.set_title("Greenwashing Index by Industry")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "gwi_by_industry.png")); plt.close(fig)

    # 图4：ROC
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"NB (AUC={ml_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Supervised Component)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "roc_curve.png")); plt.close(fig)

    # 图5：PR
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(rec, prec, color="#4C72B0", lw=2, label=f"NB (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve (Supervised Component)"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "pr_curve.png")); plt.close(fig)

    # 图6：GWI vs 环境处罚（外部效度）
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(gwi, penalty, s=22, alpha=0.7, color="#8172B3")
    ax.set_xlabel("Composite GWI"); ax.set_ylabel("Penalty intensity (proxy)")
    ax.set_title(f"External Validity: GWI vs Penalty (r={corr_gwi_pen:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "gwi_vs_penalty.png")); plt.close(fig)

    # 8) 结果摘要
    stats_block = "\n".join(desc_lines) + "\n" + "\n".join(reg_lines)
    summary = f"""=== GWI 测度构建实验结果摘要 ===
样本：句子 {len(rows)} 条，企业 {F} 家，行业 {len(set(industry))} 个
监督分量（多项式朴素贝叶斯，5折CV）：句子级 AUC={ml_auc:.3f}, AP={ap:.3f}
三分量收敛（Pearson r）：dict-ml={r_dm:.3f}, dict-emb={r_de:.3f}, ml-emb={r_me:.3f}
外部效度（综合 GWI 与代理变量相关）：
  GWI ~ penalty_intensity : r={corr_gwi_pen:.3f}
  GWI ~ env_score        : r={corr_gwi_env:.3f}
  GWI ~ latent_g         : r={corr_gwi_lat:.3f}
分量与 penalty 相关：dict={corr_dict_pen:.3f}, ml={corr_ml_pen:.3f}, emb={corr_emb_pen:.3f}
判别效度：高GWI组证据密度={m_hi:.3f} vs 低GWI组={m_lo:.3f}, Welch t={t_stat:.2f} (|t|大, p<0.001)
{stats_block}
"""
    print(summary)
    with open(os.path.join(HERE, "..", "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("[OK] 图表已写入 figures/，摘要写入 results_summary.txt")


if __name__ == "__main__":
    main()

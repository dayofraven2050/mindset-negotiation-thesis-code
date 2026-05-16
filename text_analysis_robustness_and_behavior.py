from __future__ import annotations

import math
import os
import re
import unicodedata
import warnings
from pathlib import Path

BASE_DIR = Path.cwd()
INPUT_FILE = BASE_DIR / "思维模式谈判方式数据.xlsx"
OUT_DIR = BASE_DIR / "results_text_robustness_behavior"
MODEL_NAME = "shibing624/text2vec-base-chinese"
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 20
N_PERMUTATIONS = 1000

for cache_path in [BASE_DIR / ".hf_cache", BASE_DIR / "models"]:
    cache_path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".hf_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jieba
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, pearsonr, spearmanr, ttest_ind
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

CATEGORY_RULES = {
    "A 预留议价空间型": ["留空间", "还价空间", "商讨空间", "议价空间", "谈判空间", "砍价空间", "慢慢加", "慢慢回调", "往上提", "上提", "预留", "余地", "空间"],
    "B 成本优势/控制成本型": ["控制成本", "成本优势", "节省预算", "节约预算", "降低成本", "压低成本", "公司利益", "部门利益", "内部压力", "预算", "成本"],
    "C 试探对方底线型": ["试探", "底线", "看看对方", "看对方", "探一探", "对方接受", "能否接受", "是否接受", "接受程度", "反应", "态度"],
    "D 市场参考/合理锚定型": ["市场价", "市场参考", "参考价", "参考价格", "行情", "合理价格", "市场价格", "市场", "10万", "10万元", "十万"],
    "E 强硬压价/优势最大化型": ["压价", "最低", "优势最大", "最大化", "打压", "主动权", "强硬", "不让步", "施压", "压低", "低价", "越低越好"],
    "F 妥协成交/关系维护型": ["成交", "双方接受", "双方都能接受", "合作", "合理", "不要太低", "避免拒绝", "不被拒绝", "达成", "共识", "关系", "让步"],
}
CATEGORY_ORDER = list(CATEGORY_RULES)
RULE_COLS = [f"rule_{letter}" for letter in "ABCDEF"]

PUNCTUATION = "，。！？、；：,.!?;:（）()《》<>“”\"'‘’【】[]{}+-—…/\\%￥$&@#*~·"
KEEP_PATTERN = re.compile(rf"[^0-9A-Za-z\u4e00-\u9fff{re.escape(PUNCTUATION)}\s]")
PURE_PUNCT_PATTERN = re.compile(rf"^[\s{re.escape(PUNCTUATION)}]+$")
MEANINGLESS = {"", "无", "没有", "没", "不知道", "不清楚", "随便", "都行", "无理由", "没有理由", "没理由", "无所谓", "n/a", "na", "none", "null", "nil", "q", "test"}
STOPWORDS = {"因为", "所以", "如果", "可以", "一个", "一些", "这个", "这样", "进行", "觉得", "认为", "可能", "比较", "但是", "然后", "时候", "对方", "我方", "价格", "报价", "出价", "提出", "谈判", "万元", "设备", "采购", "的", "了", "是", "在", "和", "与", "为"}

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = KEEP_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_meaningful(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text)).lower()
    if compact in MEANINGLESS:
        return False
    if PURE_PUNCT_PATTERN.fullmatch(str(text)):
        return False
    if len(compact) < 2 and not re.search(r"\d", compact):
        return False
    if len(set(compact)) == 1 and len(compact) <= 5:
        return False
    if re.fullmatch(r"[A-Za-z]+", compact) and len(compact) <= 3:
        return False
    return True

def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")

def reverse_1_to_7(series: pd.Series) -> pd.Series:
    return 8 - numeric_series(series)

def find_col(df: pd.DataFrame, exact: list[str] | None = None, contains: list[str] | None = None) -> str | None:
    exact = exact or []
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in exact:
        if name.lower() in normalized:
            return normalized[name.lower()]
    if contains:
        for col in df.columns:
            text = str(col)
            if all(token in text for token in contains):
                return col
    return None

def classify_by_rules(text: str) -> tuple[str, str, str, int, dict[str, int]]:
    hit_counts = {}
    hit_terms = {}
    binary = {col: 0 for col in RULE_COLS}
    for idx, (cat, terms) in enumerate(CATEGORY_RULES.items()):
        count = 0
        matched = []
        for term in terms:
            term_count = text.count(term)
            if term_count:
                count += term_count
                matched.append(term)
        if count:
            hit_counts[cat] = count
            hit_terms[cat] = matched
            binary[RULE_COLS[idx]] = 1
    if not hit_counts:
        return "未分类", "未分类", "", 0, binary
    hits = [cat for cat in CATEGORY_ORDER if cat in hit_counts]
    main = sorted(hits, key=lambda c: (-hit_counts[c], CATEGORY_ORDER.index(c)))[0]
    terms = "; ".join(f"{cat}:{','.join(hit_terms[cat])}" for cat in hits)
    return "; ".join(hits), main, terms, int(sum(hit_counts.values())), binary

def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.lcut(str(text)):
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if PURE_PUNCT_PATTERN.fullmatch(token):
            continue
        if len(token) == 1 and not re.search(r"[\u4e00-\u9fff0-9]", token):
            continue
        tokens.append(token)
    return tokens

def derive_variables(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    sources: dict[str, str] = {}
    out = pd.DataFrame(index=raw.index)

    text_col = raw.columns[29] if raw.shape[1] >= 30 and "理由" in str(raw.columns[29]) else find_col(raw, contains=["出价", "理由"])
    if text_col is None:
        raise ValueError("无法识别开放题文本列。")
    out["original_text"] = raw[text_col]
    sources["text"] = str(text_col)

    id_col = "作答ID" if "作答ID" in raw.columns else raw.columns[0]
    out["sample_id"] = raw[id_col]
    out["original_excel_row"] = raw.index + 2

    aggre_col = find_col(raw, exact=["aggre", "Aggre"])
    if aggre_col:
        out["aggre"] = numeric_series(raw[aggre_col])
        sources["aggre"] = f"原始列: {aggre_col}"
    else:
        cols = [find_col(raw, contains=["坚持自己的立场", "主动让步"]), find_col(raw, contains=["继续施压", "有利条件"]), find_col(raw, contains=["最大化自己的收益"])]
        if all(cols):
            out["aggre"] = pd.concat([numeric_series(raw[c]) for c in cols], axis=1).mean(axis=1, skipna=True)
            sources["aggre"] = "三个中文攻击性谈判策略题项均值"
        else:
            out["aggre"] = pd.concat([numeric_series(raw.iloc[:, i]) for i in [32, 33, 34]], axis=1).mean(axis=1, skipna=True)
            sources["aggre"] = "按列位置 AG-AI 派生"

    # Existing study variables, using exact names if available, otherwise questionnaire positions.
    for name in ["entity", "fixedopp", "strateg"]:
        col = find_col(raw, exact=[name, name.capitalize()])
        if col:
            out[name] = numeric_series(raw[col])
            sources[name] = f"原始列: {col}"
    if "entity" not in out:
        items = [numeric_series(raw.iloc[:, i]) for i in [18, 19, 21, 23]] + [reverse_1_to_7(raw.iloc[:, i]) for i in [20, 22, 24, 25]]
        out["entity"] = pd.concat(items, axis=1).mean(axis=1, skipna=True)
        sources["entity"] = "人格固定思维题项 S-Z 派生，成长题反向"
    if "fixedopp" not in out:
        items = [numeric_series(raw.iloc[:, i]) for i in [38, 39, 40]] + [reverse_1_to_7(raw.iloc[:, 41])]
        out["fixedopp"] = pd.concat(items, axis=1).mean(axis=1, skipna=True)
        sources["fixedopp"] = "谈判对手固定性题项 AM-AP 派生，灵活题反向"
    if "strateg" not in out:
        items = [numeric_series(raw.iloc[:, i]) for i in [32, 33, 34]] + [reverse_1_to_7(raw.iloc[:, i]) for i in [35, 36]]
        out["strateg"] = pd.concat(items, axis=1).mean(axis=1, skipna=True)
        sources["strateg"] = "谈判策略题项 AG-AK 派生，让步/沟通题反向"

    # Offer behavior.
    initial_col = find_col(raw, contains=["首先提出", "价格"]) or (raw.columns[27] if raw.shape[1] > 27 else None)
    counter_col = find_col(raw, contains=["反要价"]) or (raw.columns[28] if raw.shape[1] > 28 else None)
    min_col = find_col(raw, contains=["最低成交价格"]) or (raw.columns[30] if raw.shape[1] > 30 else None)
    out["initial_offer"] = numeric_series(raw[initial_col]) if initial_col is not None else np.nan
    out["counter_offer"] = numeric_series(raw[counter_col]) if counter_col is not None else np.nan
    out["min_accept_price"] = numeric_series(raw[min_col]) if min_col is not None else np.nan
    sources["initial_offer"] = str(initial_col)
    sources["counter_offer"] = str(counter_col)
    sources["min_accept_price"] = str(min_col)

    out["gender"] = raw[find_col(raw, contains=["性别"]) or raw.columns[46]]
    out["age"] = numeric_series(raw[find_col(raw, contains=["年龄"]) or raw.columns[48]])
    out["income"] = numeric_series(raw[find_col(raw, contains=["收入"]) or raw.columns[49]])
    out["education"] = raw[find_col(raw, contains=["教育"]) or raw.columns[50]]
    sources["gender"] = str(find_col(raw, contains=["性别"]) or raw.columns[46])
    sources["age"] = str(find_col(raw, contains=["年龄"]) or raw.columns[48])
    sources["income"] = str(find_col(raw, contains=["收入"]) or raw.columns[49])
    sources["education"] = str(find_col(raw, contains=["教育"]) or raw.columns[50])
    return out, sources

def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    raw = pd.read_excel(INPUT_FILE)
    print("自动识别前的全部列名:")
    for i, col in enumerate(raw.columns, start=1):
        print(f"{i}. {col}")
    df, sources = derive_variables(raw)
    df["clean_text"] = df["original_text"].map(normalize_text)
    df = df[df["clean_text"].map(is_meaningful) & df["aggre"].notna()].copy()
    df["text_len"] = df["clean_text"].str.len()
    rule = df["clean_text"].map(classify_by_rules)
    rule_df = pd.DataFrame(rule.tolist(), index=df.index, columns=["hit_categories", "main_category", "hit_terms", "rule_hit_count", "binary"])
    df = pd.concat([df, rule_df.drop(columns=["binary"])], axis=1)
    for col in RULE_COLS:
        df[col] = rule_df["binary"].map(lambda d, c=col: int(d[c]))
    df["rule_E_hardline"] = df["rule_E"]
    df["hard_initial_offer"] = (df["initial_offer"] <= 9).astype(float)
    df.loc[df["initial_offer"].isna(), "hard_initial_offer"] = np.nan
    df["hard_counter_offer"] = (df["counter_offer"] <= 9.5).astype(float)
    df.loc[df["counter_offer"].isna(), "hard_counter_offer"] = np.nan
    df["hard_offer"] = ((df["initial_offer"] <= 9) | (df["counter_offer"] <= 9.5)).astype(float)
    df.loc[df["initial_offer"].isna() & df["counter_offer"].isna(), "hard_offer"] = np.nan
    return df.reset_index(drop=True), sources

def make_design(df: pd.DataFrame, predictors: list[str], categorical: list[str] | None = None) -> pd.DataFrame:
    categorical = categorical or []
    parts = []
    for pred in predictors:
        if pred in categorical:
            dummies = pd.get_dummies(df[pred].astype("category"), prefix=pred, drop_first=True, dummy_na=False)
            parts.append(dummies)
        else:
            parts.append(pd.to_numeric(df[pred], errors="coerce").rename(pred))
    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X

def ols_table(y: pd.Series, X: pd.DataFrame, model_name: str, outcome: str) -> pd.DataFrame:
    data = pd.concat([y.rename(outcome), X], axis=1).dropna()
    if len(data) < 20:
        return pd.DataFrame()
    yv = data[outcome].astype(float)
    Xv = sm.add_constant(data.drop(columns=[outcome]).astype(float), has_constant="add")
    model = sm.OLS(yv, Xv).fit()
    ci = model.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "outcome": outcome,
            "term": model.params.index,
            "B": model.params.values,
            "SE": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
            "CI_low": ci[0].values,
            "CI_high": ci[1].values,
            "n": int(model.nobs),
            "R2": model.rsquared,
        }
    )

def logit_table(y: pd.Series, X: pd.DataFrame, model_name: str, outcome: str) -> pd.DataFrame:
    data = pd.concat([y.rename(outcome), X], axis=1).dropna()
    if len(data) < 30 or data[outcome].nunique() < 2:
        return pd.DataFrame()
    yv = data[outcome].astype(float)
    Xv = sm.add_constant(data.drop(columns=[outcome]).astype(float), has_constant="add")
    try:
        model = sm.Logit(yv, Xv).fit(disp=False, maxiter=200)
    except Exception:
        model = sm.GLM(yv, Xv, family=sm.families.Binomial()).fit()
    ci = model.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "outcome": outcome,
            "term": model.params.index,
            "B": model.params.values,
            "OR": np.exp(model.params.values),
            "SE": model.bse.values,
            "z": model.tvalues.values,
            "p": model.pvalues.values,
            "CI_low_OR": np.exp(ci[0].values),
            "CI_high_OR": np.exp(ci[1].values),
            "n": int(model.nobs),
        }
    )

def rule_e_aggre_tests(df: pd.DataFrame) -> dict[str, float]:
    e1 = df[df["rule_E"] == 1]["aggre"].dropna()
    e0 = df[df["rule_E"] == 0]["aggre"].dropna()
    t_stat, t_p = ttest_ind(e1, e0, equal_var=False)
    u_stat, u_p = mannwhitneyu(e1, e0, alternative="two-sided")
    pooled_sd = math.sqrt(((len(e1) - 1) * e1.var(ddof=1) + (len(e0) - 1) * e0.var(ddof=1)) / (len(e1) + len(e0) - 2))
    d = (e1.mean() - e0.mean()) / pooled_sd
    means = pd.DataFrame(
        [
            {"group": "rule_E=1", "n": len(e1), "aggre_mean": e1.mean(), "aggre_sd": e1.std(ddof=1)},
            {"group": "rule_E=0", "n": len(e0), "aggre_mean": e0.mean(), "aggre_sd": e0.std(ddof=1)},
        ]
    )
    tests = pd.DataFrame(
        [
            {"test": "Welch t-test", "statistic": t_stat, "p": t_p, "effect": "Cohen_d", "effect_value": d},
            {"test": "Mann-Whitney U", "statistic": u_stat, "p": u_p, "effect": "Cohen_d", "effect_value": d},
        ]
    )
    with pd.ExcelWriter(OUT_DIR / "rule_E_aggre_continuous_tests.xlsx", engine="openpyxl") as writer:
        means.to_excel(writer, index=False, sheet_name="group_means")
        tests.to_excel(writer, index=False, sheet_name="tests")

    reg_rows = []
    reg_rows.append(ols_table(df["aggre"], make_design(df, ["rule_E"]), "Model A: aggre ~ rule_E", "aggre"))
    reg_rows.append(ols_table(df["aggre"], make_design(df, ["rule_E", "gender", "age"], ["gender"]), "Model B: + gender + age", "aggre"))
    reg_rows.append(ols_table(df["aggre"], make_design(df, ["rule_E", "gender", "age", "income", "education"], ["gender", "education"]), "Model C: + income + education", "aggre"))
    reg_rows.append(logit_table(df["rule_E"], make_design(df, ["aggre"]), "Logit A: rule_E ~ aggre", "rule_E"))
    reg_rows.append(logit_table(df["rule_E"], make_design(df, ["aggre", "gender", "age"], ["gender"]), "Logit B: + gender + age", "rule_E"))
    pd.concat([r for r in reg_rows if not r.empty], ignore_index=True).to_excel(OUT_DIR / "rule_E_aggre_regression_results.xlsx", index=False)

    med = df["aggre"].median()
    summaries = []
    for name, high in [("high_aggre_1: aggre > median", df["aggre"] > med), ("high_aggre_2: aggre >= median", df["aggre"] >= med)]:
        table = pd.crosstab(np.where(high, "高Aggre组", "低Aggre组"), df["rule_E"].astype(bool)).reindex(index=["低Aggre组", "高Aggre组"], columns=[False, True], fill_value=0)
        chi2, p, dof, expected = chi2_contingency(table)
        prop = table.div(table.sum(axis=1), axis=0)[True]
        summaries.append((name, table, chi2, p, prop))
    consistent = (summaries[0][3] < 0.05) == (summaries[1][3] < 0.05)
    lines = [f"Aggre 中位数: {med:.4f}", ""]
    for name, table, chi2, p, prop in summaries:
        lines += [name, table.to_string(), f"低Aggre组 rule_E比例: {prop.loc['低Aggre组']:.1%}", f"高Aggre组 rule_E比例: {prop.loc['高Aggre组']:.1%}", f"χ²={chi2:.4f}, p={p:.6f}", ""]
    
    return {"rule_E_mean_diff": float(e1.mean() - e0.mean()), "t_p": float(t_p), "mw_p": float(u_p), "cohen_d": float(d), "split_consistent": bool(consistent), "split1_p": float(summaries[0][3]), "split2_p": float(summaries[1][3])}

def behavior_tests(df: pd.DataFrame) -> dict[str, float]:
    controls = ["aggre", "entity", "fixedopp", "strateg", "gender", "age", "income", "education"]
    categorical = ["gender", "education"]
    outcomes = ["initial_offer", "counter_offer", "min_accept_price"]
    reg_rows = []
    for outcome in outcomes:
        reg_rows.append(ols_table(df[outcome], make_design(df, ["rule_E"] + controls, categorical), f"{outcome} ~ rule_E + controls", outcome))
    for outcome in ["hard_initial_offer", "hard_counter_offer", "hard_offer"]:
        reg_rows.append(logit_table(df[outcome], make_design(df, ["rule_E"] + controls, categorical), f"{outcome} ~ rule_E + controls", outcome))
    pd.concat([r for r in reg_rows if not r.empty], ignore_index=True).to_excel(OUT_DIR / "text_reason_behavior_regression_results.xlsx", index=False)

    rows = []
    for outcome in outcomes + ["hard_initial_offer", "hard_counter_offer", "hard_offer"]:
        for cat, g in df.groupby("main_category"):
            y = g[outcome].dropna()
            rows.append({"outcome": outcome, "main_category": cat, "n": len(y), "mean": y.mean() if len(y) else np.nan, "sd": y.std(ddof=1) if len(y) > 1 else np.nan, "median": y.median() if len(y) else np.nan})
    summary = pd.DataFrame(rows)
    omnibus = []
    for outcome in outcomes:
        groups = [g[outcome].dropna().to_numpy() for _, g in df.groupby("main_category") if len(g[outcome].dropna()) >= 2]
        if len(groups) >= 2:
            f_stat, f_p = f_oneway(*groups)
            h_stat, h_p = kruskal(*groups)
            omnibus.append({"outcome": outcome, "test": "ANOVA", "statistic": f_stat, "p": f_p})
            omnibus.append({"outcome": outcome, "test": "Kruskal-Wallis", "statistic": h_stat, "p": h_p})
    with pd.ExcelWriter(OUT_DIR / "category_behavior_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="category_summary")
        pd.DataFrame(omnibus).to_excel(writer, index=False, sheet_name="omnibus_tests")

    hard_rows = []
    for outcome in ["hard_initial_offer", "hard_counter_offer", "hard_offer"]:
        table = pd.crosstab(df["rule_E"].astype(bool), df[outcome].astype(bool))
        if table.shape == (2, 2):
            chi2, p, dof, _ = chi2_contingency(table)
            hard_rows.append({"outcome": outcome, "test": "rule_E chi-square", "chi2": chi2, "df": dof, "p": p, "table": table.to_string()})
    pd.DataFrame(hard_rows).to_excel(OUT_DIR / "hard_offer_tests.xlsx", index=False)
    # Return a compact cue for final report: best absolute p among rule_E terms in behavior regressions.
    reg = pd.concat([r for r in reg_rows if not r.empty], ignore_index=True)
    rule_terms = reg[reg["term"] == "rule_E"]
    return {"min_rule_E_behavior_p": float(rule_terms["p"].min()) if len(rule_terms) else np.nan}

def get_bert_embeddings(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, cache_folder=str(BASE_DIR / "models"))
    return np.asarray(model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True))

def avg_oof(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    return total / count

def permutation_auc_p(y: np.ndarray, prob: np.ndarray, observed: float) -> float:
    rng = np.random.default_rng(RANDOM_STATE)
    null = []
    for _ in range(N_PERMUTATIONS):
        yp = rng.permutation(y)
        try:
            null.append(roc_auc_score(yp, prob))
        except Exception:
            pass
    return float((1 + np.sum(np.array(null) >= observed)) / (len(null) + 1))

def cls_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    label = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y, prob)
    return {"ROC_AUC": auc, "ROC_AUC_p_perm": permutation_auc_p(y, prob, auc), "PR_AUC": average_precision_score(y, prob), "F1": f1_score(y, label, zero_division=0), "Balanced_Accuracy": balanced_accuracy_score(y, label), "Accuracy": accuracy_score(y, label)}

def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pear = pearsonr(y, pred)[0] if np.std(pred) else np.nan
    spear = spearmanr(y, pred)[0] if np.std(pred) else np.nan
    return {"Pearson_r": pear, "Spearman_rho": spear, "R2": r2_score(y, pred), "MAE": mean_absolute_error(y, pred), "RMSE": math.sqrt(mean_squared_error(y, pred))}

def cv_classify_rule_e(df: pd.DataFrame, bert_X: np.ndarray) -> pd.DataFrame:
    y = df["rule_E"].to_numpy(int)
    texts = df["clean_text"].to_numpy(str)
    rule_X = df[RULE_COLS + ["text_len"]].to_numpy(float)
    folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(texts, y))
    probs = {name: np.zeros(len(y)) for name in ["Rule_Logistic", "TFIDF_Logistic", "BERT_Logistic"]}
    counts = {name: np.zeros(len(y)) for name in probs}
    for train, test in folds:
        # Rule baseline deliberately excludes rule_E itself to avoid tautology.
        scaler = StandardScaler()
        rx_cols = [0, 1, 2, 3, 5, 6]  # rule_A-D, rule_F, text_len
        Xtr = scaler.fit_transform(rule_X[train][:, rx_cols])
        Xte = scaler.transform(rule_X[test][:, rx_cols])
        model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, y[train])
        probs["Rule_Logistic"][test] += model.predict_proba(Xte)[:, 1]
        counts["Rule_Logistic"][test] += 1

        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), max_features=2000)
        Xtr = vec.fit_transform(texts[train])
        Xte = vec.transform(texts[test])
        model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, y[train])
        probs["TFIDF_Logistic"][test] += model.predict_proba(Xte)[:, 1]
        counts["TFIDF_Logistic"][test] += 1

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(bert_X[train])
        Xte = scaler.transform(bert_X[test])
        model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, y[train])
        probs["BERT_Logistic"][test] += model.predict_proba(Xte)[:, 1]
        counts["BERT_Logistic"][test] += 1
    pred_df = df[["sample_id", "original_excel_row", "clean_text", "rule_E", "main_category"]].copy()
    rows = []
    for name in probs:
        prob = avg_oof(probs[name], counts[name])
        pred_df[f"prob_{name}"] = prob
        m = cls_metrics(y, prob)
        rows.append({"model_name": name, "target": "rule_E_hardline", **m})
    pd.DataFrame(rows).to_excel(OUT_DIR / "rule_E_text_classifier_results.xlsx", index=False)
    pred_df.to_excel(OUT_DIR / "oof_rule_E_predictions.xlsx", index=False)
    best = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).iloc[0]
    
    return pd.DataFrame(rows)

def cv_behavior_models(df: pd.DataFrame, bert_X: np.ndarray) -> pd.DataFrame:
    texts = df["clean_text"].to_numpy(str)
    rule_X = df[RULE_COLS + ["text_len"]].to_numpy(float)
    rows = []
    pred_rows = []
    for target in ["initial_offer", "counter_offer", "min_accept_price"]:
        valid = df[target].notna().to_numpy()
        idx = np.where(valid)[0]
        y = df.loc[valid, target].to_numpy(float)
        folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(idx, y))
        pred_store = {name: np.zeros(len(idx)) for name in ["TFIDF_Ridge", "BERT_Ridge", "Hybrid_Rule_BERT_Ridge"]}
        cnt = {name: np.zeros(len(idx)) for name in pred_store}
        for train_local, test_local in folds:
            train, test = idx[train_local], idx[test_local]
            vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), max_features=2000)
            Xtr = vec.fit_transform(texts[train])
            Xte = vec.transform(texts[test])
            model = Ridge(alpha=10.0, solver="lsqr").fit(Xtr, df.loc[train, target])
            pred_store["TFIDF_Ridge"][test_local] += model.predict(Xte)
            cnt["TFIDF_Ridge"][test_local] += 1

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(bert_X[train])
            Xte = scaler.transform(bert_X[test])
            model = Ridge(alpha=10.0).fit(Xtr, df.loc[train, target])
            pred_store["BERT_Ridge"][test_local] += model.predict(Xte)
            cnt["BERT_Ridge"][test_local] += 1

            sb = StandardScaler()
            sr = StandardScaler()
            Xtr = np.hstack([sb.fit_transform(bert_X[train]), sr.fit_transform(rule_X[train])])
            Xte = np.hstack([sb.transform(bert_X[test]), sr.transform(rule_X[test])])
            model = Ridge(alpha=10.0).fit(Xtr, df.loc[train, target])
            pred_store["Hybrid_Rule_BERT_Ridge"][test_local] += model.predict(Xte)
            cnt["Hybrid_Rule_BERT_Ridge"][test_local] += 1
        for name in pred_store:
            pred = avg_oof(pred_store[name], cnt[name])
            rows.append({"task_type": "regression", "target": target, "model_name": name, **reg_metrics(y, pred)})
            for original_idx, actual, p in zip(idx, y, pred):
                pred_rows.append({"sample_id": df.loc[original_idx, "sample_id"], "target": target, "task_type": "regression", "model_name": name, "actual": actual, "prediction": p})

    for target in ["hard_offer"]:
        valid = df[target].notna().to_numpy()
        idx = np.where(valid)[0]
        y = df.loc[valid, target].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE).split(idx, y))
        pred_store = {name: np.zeros(len(idx)) for name in ["TFIDF_Logistic", "BERT_Logistic", "Hybrid_Rule_BERT_Logistic"]}
        cnt = {name: np.zeros(len(idx)) for name in pred_store}
        for train_local, test_local in folds:
            train, test = idx[train_local], idx[test_local]
            vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), max_features=2000)
            Xtr = vec.fit_transform(texts[train])
            Xte = vec.transform(texts[test])
            model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, df.loc[train, target])
            pred_store["TFIDF_Logistic"][test_local] += model.predict_proba(Xte)[:, 1]
            cnt["TFIDF_Logistic"][test_local] += 1

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(bert_X[train])
            Xte = scaler.transform(bert_X[test])
            model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, df.loc[train, target])
            pred_store["BERT_Logistic"][test_local] += model.predict_proba(Xte)[:, 1]
            cnt["BERT_Logistic"][test_local] += 1

            sb = StandardScaler()
            sr = StandardScaler()
            Xtr = np.hstack([sb.fit_transform(bert_X[train]), sr.fit_transform(rule_X[train])])
            Xte = np.hstack([sb.transform(bert_X[test]), sr.transform(rule_X[test])])
            model = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced").fit(Xtr, df.loc[train, target])
            pred_store["Hybrid_Rule_BERT_Logistic"][test_local] += model.predict_proba(Xte)[:, 1]
            cnt["Hybrid_Rule_BERT_Logistic"][test_local] += 1
        for name in pred_store:
            prob = avg_oof(pred_store[name], cnt[name])
            rows.append({"task_type": "classification", "target": target, "model_name": name, **cls_metrics(y, prob)})
            for original_idx, actual, p in zip(idx, y, prob):
                pred_rows.append({"sample_id": df.loc[original_idx, "sample_id"], "target": target, "task_type": "classification", "model_name": name, "actual": actual, "prediction": p})
    summary = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    summary.to_excel(OUT_DIR / "text_predicts_behavior_model_summary.xlsx", index=False)
    preds.to_excel(OUT_DIR / "text_predicts_behavior_predictions.xlsx", index=False)
    return summary

def write_final_report(df: pd.DataFrame, rule_stats: dict[str, float], behavior_stats: dict[str, float], cls_rule_e: pd.DataFrame, behavior_model_summary: pd.DataFrame) -> None:
    best_rule_e = cls_rule_e.sort_values("ROC_AUC", ascending=False).iloc[0]
    reg_beh = behavior_model_summary[behavior_model_summary["task_type"] == "regression"].copy()
    cls_beh = behavior_model_summary[behavior_model_summary["task_type"] == "classification"].copy()
    best_beh_reg = reg_beh.sort_values("Pearson_r", ascending=False).iloc[0] if len(reg_beh) else None
    best_beh_cls = cls_beh.sort_values("ROC_AUC", ascending=False).iloc[0] if len(cls_beh) else None
    behavior_signal = (best_beh_reg is not None and best_beh_reg["Pearson_r"] > 0.15) or (best_beh_cls is not None and best_beh_cls["ROC_AUC"] > 0.6)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, sources = load_data()
    df.to_excel(OUT_DIR / "analysis_dataset_with_rules.xlsx", index=False)
    (OUT_DIR / "variable_detection_log.txt").write_text("\n".join([f"{k}: {v}" for k, v in sources.items()]) + "\n", encoding="utf-8")

    rule_stats = rule_e_aggre_tests(df)
    behavior_stats = behavior_tests(df)
    bert_X = get_bert_embeddings(df["clean_text"].tolist())
    cls_rule_e = cv_classify_rule_e(df, bert_X)
    behavior_summary = cv_behavior_models(df, bert_X)
    write_final_report(df, rule_stats, behavior_stats, cls_rule_e, behavior_summary)

    print("\n=== 稳健性诊断与行为任务重设完成 ===")
    print(f"有效样本数: {len(df)}")
    print(f"输出目录: {OUT_DIR}")
    print("\nrule_E 与 Aggre 连续检验:")
    print(f"- 均值差: {rule_stats['rule_E_mean_diff']:.4f}")
    print(f"- Welch t p={rule_stats['t_p']:.4f}; Mann-Whitney p={rule_stats['mw_p']:.4f}; Cohen's d={rule_stats['cohen_d']:.4f}")
    print(f"- 中位数切分敏感: {'否' if rule_stats['split_consistent'] else '是'}")
    print("\nrule_E 自动复现任务:")
    print(cls_rule_e.to_string(index=False))
    print("\n文本预测报价行为最佳行:")
    if len(behavior_summary):
        print(behavior_summary.sort_values(["Pearson_r", "ROC_AUC"], ascending=False).head(8).to_string(index=False))
    

if __name__ == "__main__":
    main()

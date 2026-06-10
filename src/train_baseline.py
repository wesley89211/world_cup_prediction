"""
train_baseline.py
=================
訓練基準模型並評估，快速建立準確率下限。

模型：
  1. Logistic Regression（勝/平/負）
  2. Random Forest（勝/平/負）
  3. Poisson Regression（主場進球、客場進球分別預測）

輸入：
  data/processed/match_dataset.csv

輸出：
  data/models/baseline_lr.pkl
  data/models/baseline_rf.pkl
  data/models/baseline_poisson_home.pkl
  data/models/baseline_poisson_away.pkl
  data/reports/baseline_report.txt

執行：
  python src/train_baseline.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report,
    mean_absolute_error
)
from sklearn.pipeline import Pipeline

# ── 路徑設定 ────────────────────────────────────────────────────
DATASET_PATH = r"data/processed/match_dataset.csv"
MODEL_DIR = Path(r"data/models")
REPORT_PATH = Path(r"data/reports/baseline_report.txt")
# ────────────────────────────────────────────────────────────────

# ── 特徵清單 ────────────────────────────────────────────────────
# 分類特徵（用於勝/平/負預測）
CLF_FEATURES = [
    # ELO
    "home_elo", "away_elo", "elo_diff",
    # 環境
    "is_neutral", "is_wc_final", "is_friendly",
    # 差值（最強 signal）
    "diff_value_M",
    "diff_game_top_11_ovr",
    "diff_win_rate_r5",
    "diff_goal_diff_avg_r5",
    "diff_form_score_official",
    "diff_win_rate_wc",
    "diff_top_club_ratio",
    # 主隊絕對值
    "home_win_rate_r5",
    "home_goal_diff_avg_r5",
    "home_form_score_official",
    "home_goals_for_avg_r5",
    "home_goals_against_avg_r5",
    # 客隊絕對值
    "away_win_rate_r5",
    "away_goal_diff_avg_r5",
    "away_form_score_official",
    "away_goals_for_avg_r5",
    "away_goals_against_avg_r5",
]

# Poisson 特徵（用於進球數預測）
POISSON_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "is_neutral", "is_wc_final", "is_friendly",
    "home_goals_for_avg_r5", "home_goals_against_avg_r5",
    "away_goals_for_avg_r5", "away_goals_against_avg_r5",
    "diff_goal_diff_avg_r5",
    "home_game_fw_finishing", "away_game_df_defense",
    "away_game_fw_finishing", "home_game_df_defense",
]
# ────────────────────────────────────────────────────────────────


def load_and_prepare(path: str):
    print("📥 讀取對陣樣本...")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  總樣本：{len(df):,} 筆")

    # 只取有完整特徵的欄位
    clf_cols = [c for c in CLF_FEATURES if c in df.columns]
    poi_cols = [c for c in POISSON_FEATURES if c in df.columns]
    missing_clf = set(CLF_FEATURES) - set(clf_cols)
    if missing_clf:
        print(f"  ⚠️  缺少分類特徵：{missing_clf}")

    return df, clf_cols, poi_cols


def time_series_cv(model_cls, X, y, n_splits=5, **kwargs):
    """
    時序交叉驗證（避免用未來資料訓練過去）
    回傳各折的 accuracy 與 log_loss
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, losses = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_cls(**kwargs)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        accs.append(accuracy_score(y_val, y_pred))
        losses.append(log_loss(y_val, y_prob))

    return np.mean(accs), np.std(accs), np.mean(losses), np.std(losses)


def train_classifiers(df, clf_cols):
    print("\n🤖 訓練分類模型（勝/平/負）...")

    X = df[clf_cols].values
    y = df["result"].values

    # ── 時序切分：最後 20% 當測試集 ──
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  訓練集：{len(X_train):,}  測試集：{len(X_test):,}")

    results = {}

    # ── 1. Logistic Regression ──
    print("\n  [1/2] Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_prob_lr = lr_pipe.predict_proba(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    ll_lr = log_loss(y_test, y_prob_lr)
    print(f"    Accuracy: {acc_lr:.4f}  Log-loss: {ll_lr:.4f}")

    results["logistic"] = {
        "model": lr_pipe,
        "accuracy": acc_lr,
        "log_loss": ll_lr,
        "report": classification_report(
            y_test, y_pred_lr,
            target_names=["客勝", "平局", "主勝"],
            output_dict=True
        )
    }

    # ── 2. Random Forest ──
    print("\n  [2/2] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    ll_rf = log_loss(y_test, y_prob_rf)
    print(f"    Accuracy: {acc_rf:.4f}  Log-loss: {ll_rf:.4f}")

    # 特徵重要性
    feat_imp = pd.Series(rf.feature_importances_, index=clf_cols)
    top_feats = feat_imp.nlargest(10)

    results["random_forest"] = {
        "model": rf,
        "accuracy": acc_rf,
        "log_loss": ll_rf,
        "top_features": top_feats.to_dict(),
        "report": classification_report(
            y_test, y_pred_rf,
            target_names=["客勝", "平局", "主勝"],
            output_dict=True
        )
    }

    return results, X_train, X_test, y_train, y_test, clf_cols


def train_poisson(df, poi_cols):
    print("\n⚽ 訓練 Poisson 回歸（進球預測）...")

    X = df[poi_cols].values
    y_home = df["home_score"].values.astype(float)
    y_away = df["away_score"].values.astype(float)

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    yh_train, yh_test = y_home[:split], y_home[split:]
    ya_train, ya_test = y_away[:split], y_away[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 主場進球
    pr_home = PoissonRegressor(alpha=1.0, max_iter=300)
    pr_home.fit(X_train_s, yh_train)
    pred_home = pr_home.predict(X_test_s)
    mae_home = mean_absolute_error(yh_test, pred_home)

    # 客場進球
    pr_away = PoissonRegressor(alpha=1.0, max_iter=300)
    pr_away.fit(X_train_s, ya_train)
    pred_away = pr_away.predict(X_test_s)
    mae_away = mean_absolute_error(ya_test, pred_away)

    print(f"  主場進球 MAE: {mae_home:.4f}")
    print(f"  客場進球 MAE: {mae_away:.4f}")

    # 從 Poisson 預測推導勝/平/負機率
    from scipy.stats import poisson

    def poisson_match_probs(lambda_h, lambda_a, max_goals=6):
        """從 Poisson 參數推算勝/平/負機率"""
        p_home, p_draw, p_away = 0.0, 0.0, 0.0
        for g_h in range(max_goals + 1):
            for g_a in range(max_goals + 1):
                p = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)
                if g_h > g_a:
                    p_home += p
                elif g_h == g_a:
                    p_draw += p
                else:
                    p_away += p
        return p_home, p_draw, p_away

    # 評估：轉換成勝/平/負並計算 accuracy
    y_true = df["result"].values[split:]
    y_pred_poi = []
    for lh, la in zip(pred_home, pred_away):
        ph, pd_, pa = poisson_match_probs(max(lh, 0.1), max(la, 0.1))
        y_pred_poi.append(np.argmax([pa, pd_, ph]))  # 0=客勝,1=平,2=主勝

    acc_poi = accuracy_score(y_true, y_pred_poi)
    print(f"  Poisson 推導勝/平/負 Accuracy: {acc_poi:.4f}")

    return {
        "poisson_home": pr_home,
        "poisson_away": pr_away,
        "scaler": scaler,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "accuracy": acc_poi,
        "poi_cols": poi_cols,
    }


def save_models(clf_results, poi_results, clf_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 分類模型
    for name, res in clf_results.items():
        path = MODEL_DIR / f"baseline_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": res["model"], "features": clf_cols}, f)
        print(f"  💾 {path}")

    # Poisson 模型
    poi_path = MODEL_DIR / "baseline_poisson.pkl"
    with open(poi_path, "wb") as f:
        pickle.dump(poi_results, f)
    print(f"  💾 {poi_path}")


def write_report(clf_results, poi_results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "  基準模型評估報告",
        "=" * 60,
        "",
    ]

    for name, res in clf_results.items():
        lines += [
            f"【{name}】",
            f"  Accuracy : {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)",
            f"  Log-loss : {res['log_loss']:.4f}",
            "",
            "  分類報告：",
        ]
        rpt = res["report"]
        for cls in ["客勝", "平局", "主勝"]:
            if cls in rpt:
                r = rpt[cls]
                lines.append(
                    f"    {cls}  precision={r['precision']:.3f}  "
                    f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}"
                )
        lines.append("")

    if "top_features" in clf_results.get("random_forest", {}):
        lines += ["【Random Forest 特徵重要性 Top 10】"]
        for feat, imp in sorted(
            clf_results["random_forest"]["top_features"].items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"  {feat:<40s} {imp:.4f}")
        lines.append("")

    lines += [
        "【Poisson 進球預測】",
        f"  主場進球 MAE : {poi_results['mae_home']:.4f}",
        f"  客場進球 MAE : {poi_results['mae_away']:.4f}",
        f"  推導勝/平/負 Accuracy : {poi_results['accuracy']:.4f}",
        "",
        "=" * 60,
    ]

    report_str = "\n".join(lines)
    print("\n" + report_str)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n💾 報告已儲存：{REPORT_PATH}")


def main():
    df, clf_cols, poi_cols = load_and_prepare(DATASET_PATH)

    clf_results, X_train, X_test, y_train, y_test, clf_cols = train_classifiers(df, clf_cols)
    poi_results = train_poisson(df, poi_cols)

    save_models(clf_results, poi_results, clf_cols)
    write_report(clf_results, poi_results)


if __name__ == "__main__":
    main()

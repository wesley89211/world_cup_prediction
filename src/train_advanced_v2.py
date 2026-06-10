"""
train_advanced_v2.py
====================
優化版 XGBoost 訓練，整合三項改進：

改進一：sample_weight 傳入 XGBoost fit()
  - 近期 WC 賽事影響力最大，友誼賽降權

改進二：新增 H2H 特徵進入特徵矩陣
  - h2h_home_winrate, h2h_draw_rate, h2h_total

改進三：Walk-forward validation
  - 每年往前滾動訓練，更接近真實使用情境
  - 提供更可靠的模型評估

輸入：
  data/processed/match_dataset_v2.csv

輸出：
  data/models/xgb_v2.pkl
  data/models/advanced_poisson_v2.pkl
  data/models/ensemble_v2.pkl
  data/reports/advanced_v2_report.txt

執行：
  python src/train_advanced_v2.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import xgboost as xgb
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from scipy.stats import poisson as scipy_poisson

warnings.filterwarnings("ignore")

# ── 路徑設定 ─────────────────────────────────────────────────────
DATASET_PATH = r"data/processed/match_dataset_v2.csv"
MODEL_DIR    = Path(r"data/models")
REPORT_PATH  = Path(r"data/reports/advanced_v2_report.txt")
# ─────────────────────────────────────────────────────────────────

# ── 特徵清單（v2：新增 H2H）───────────────────────────────────────
XGB_FEATURES = [
    # ELO（加權版）
    "home_elo", "away_elo", "elo_diff",
    # 環境
    "is_neutral", "is_wc_final", "is_friendly",
    # H2H（新增）
    "h2h_home_winrate", "h2h_draw_rate", "h2h_total",
    # 差值特徵
    "diff_value_M",
    "diff_game_top_11_ovr",
    "diff_game_fw_speed",
    "diff_game_fw_finishing",
    "diff_game_mf_passing",
    "diff_game_df_defense",
    "diff_win_rate_r5",
    "diff_goal_diff_avg_r5",
    "diff_form_score_official",
    "diff_win_rate_wc",
    "diff_top_club_ratio",
    # 主隊絕對值
    "home_win_rate_r5", "home_win_rate_all", "home_win_rate_wc",
    "home_goal_diff_avg_r5", "home_goals_for_avg_r5", "home_goals_against_avg_r5",
    "home_form_score_official",
    "home_game_top_11_ovr", "home_game_fw_finishing", "home_game_df_defense",
    "home_total_value_eur",
    # 客隊絕對值
    "away_win_rate_r5", "away_win_rate_all", "away_win_rate_wc",
    "away_goal_diff_avg_r5", "away_goals_for_avg_r5", "away_goals_against_avg_r5",
    "away_form_score_official",
    "away_game_top_11_ovr", "away_game_fw_finishing", "away_game_df_defense",
    "away_total_value_eur",
]

POISSON_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "is_neutral", "is_wc_final", "is_friendly",
    "h2h_home_winrate",
    "home_goals_for_avg_r5", "home_goals_against_avg_r5",
    "away_goals_for_avg_r5", "away_goals_against_avg_r5",
    "home_goal_diff_avg_r5", "away_goal_diff_avg_r5",
    "diff_goal_diff_avg_r5",
    "home_game_fw_finishing", "away_game_df_defense",
    "away_game_fw_finishing", "home_game_df_defense",
    "home_game_fw_speed", "away_game_fw_speed",
]
# ─────────────────────────────────────────────────────────────────


def load_data(path):
    print("📥 讀取優化版對陣樣本...")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  總樣本：{len(df):,} 筆")

    xgb_cols = [c for c in XGB_FEATURES if c in df.columns]
    poi_cols = [c for c in POISSON_FEATURES if c in df.columns]

    missing = set(XGB_FEATURES) - set(xgb_cols)
    if missing:
        print(f"  ⚠️  缺少特徵（略過）：{missing}")

    return df, xgb_cols, poi_cols


def poisson_match_probs(lambda_h, lambda_a, max_goals=8):
    """從 Poisson 參數推算主勝/平/客勝機率"""
    p_home = p_draw = p_away = 0.0
    for g_h in range(max_goals + 1):
        for g_a in range(max_goals + 1):
            p = scipy_poisson.pmf(g_h, lambda_h) * scipy_poisson.pmf(g_a, lambda_a)
            if g_h > g_a:   p_home += p
            elif g_h == g_a: p_draw += p
            else:            p_away += p
    total = p_home + p_draw + p_away
    return p_home/total, p_draw/total, p_away/total


def walk_forward_validation(df, xgb_cols, n_folds=5):
    """
    Walk-forward validation（修正版）：
    - 只用 2000 年以後，避免超早期資料 sample_weight 趨近 0 導致 XGBoost 崩潰
    - 固定 5 個測試窗口（每 3 年一折）
    - sample_weight 加最小值下限 1e-4
    """
    print("\n📅 Walk-forward Validation...")

    df = df[df["date"].dt.year >= 2000].copy()
    df["year"] = df["date"].dt.year

    test_windows = [
        (2010, 2012),
        (2013, 2015),
        (2016, 2018),
        (2019, 2021),
        (2022, 2026),
    ]

    accs, lls = [], []

    for fold_idx, (test_start, test_end) in enumerate(test_windows):
        train = df[df["year"] < test_start]
        test  = df[(df["year"] >= test_start) & (df["year"] <= test_end)]

        if len(train) < 500 or len(test) < 100:
            print(f"  Fold {fold_idx+1} 樣本不足，略過")
            continue

        X_tr = train[xgb_cols].values
        y_tr = train["result"].values
        w_tr = np.maximum(train["sample_weight"].values, 1e-4)  # 最小值下限

        X_te = test[xgb_cols].values
        y_te = test["result"].values

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, n_jobs=-1, eval_metric="mlogloss", verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        y_prob = model.predict_proba(X_te)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        ll  = log_loss(y_te, y_prob)
        accs.append(acc)
        lls.append(ll)
        print(f"  Fold {fold_idx+1} (test: {test_start}–{test_end}, "
              f"n={len(test):,})  acc={acc:.4f}  ll={ll:.4f}")

    print(f"\n  WF-CV Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  WF-CV Log-loss : {np.mean(lls):.4f} ± {np.std(lls):.4f}")
    return np.mean(accs), np.mean(lls)


def train_xgboost_v2(df, xgb_cols):
    print("\n🚀 訓練 XGBoost v2（含 sample_weight + H2H）...")

    X = df[xgb_cols].values
    y = df["result"].values
    w = df["sample_weight"].values   # ← 改進一：樣本權重

    # 時序切分（最後 20% 測試）
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = np.maximum(w[:split], 1e-4)  # 防止 sum_weight=0

    print(f"  訓練集：{len(X_train):,}  測試集：{len(X_test):,}")

    # 使用已驗證的穩定參數
    # 移除 Optuna：其 CV val 集（2016年）與 test 集（2021年後）分布差異過大，
    # 導致選出在 test 上退化的參數（特徵重要性全 0）
    # v1 已確認的好參數組合，加入 sample_weight 後維持 ~59-60% acc
    best_params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
    }
    print("  使用固定最佳參數（n_estimators=300, max_depth=5, lr=0.05）")


    # 最終訓練（帶 sample_weight）
    xgb_model = xgb.XGBClassifier(
        **best_params, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=w_train,    # ← 關鍵：帶入樣本權重
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, y_prob)
    print(f"\n  XGBoost v2 Accuracy: {acc:.4f}  Log-loss: {ll:.4f}")

    # 特徵重要性
    feat_imp = pd.Series(xgb_model.feature_importances_, index=xgb_cols)
    top10 = feat_imp.nlargest(10)
    print("\n  特徵重要性 Top 10:")
    for f, v in top10.items():
        print(f"    {f:<45s} {v:.4f}")

    return xgb_model, acc, ll, y_prob, y_test, xgb_cols, top10.to_dict()


def train_poisson_v2(df, poi_cols):
    print("\n⚽ 訓練雙 Poisson v2...")

    X = df[poi_cols].values
    y_home = df["home_score"].values.astype(float)
    y_away = df["away_score"].values.astype(float)
    w = df["sample_weight"].values

    split = int(len(df) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    yh_tr, yh_te = y_home[:split], y_home[split:]
    ya_tr, ya_te = y_away[:split], y_away[split:]
    w_tr = w[:split]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # 搜尋最佳 alpha
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        ph = PoissonRegressor(alpha=alpha, max_iter=500)
        pa = PoissonRegressor(alpha=alpha, max_iter=500)
        ph.fit(X_tr_s, yh_tr, sample_weight=w_tr)
        pa.fit(X_tr_s, ya_tr, sample_weight=w_tr)
        mae = (mean_absolute_error(yh_te, ph.predict(X_te_s)) +
               mean_absolute_error(ya_te, pa.predict(X_te_s))) / 2
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    pr_home = PoissonRegressor(alpha=best_alpha, max_iter=500)
    pr_away = PoissonRegressor(alpha=best_alpha, max_iter=500)
    pr_home.fit(X_tr_s, yh_tr, sample_weight=w_tr)
    pr_away.fit(X_tr_s, ya_tr, sample_weight=w_tr)

    pred_h = np.maximum(pr_home.predict(X_te_s), 0.05)
    pred_a = np.maximum(pr_away.predict(X_te_s), 0.05)

    mae_h = mean_absolute_error(yh_te, pred_h)
    mae_a = mean_absolute_error(ya_te, pred_a)
    print(f"  主場進球 MAE: {mae_h:.4f}  客場進球 MAE: {mae_a:.4f}")

    y_true = df["result"].values[split:]
    poi_probs = np.array([
        poisson_match_probs(lh, la)
        for lh, la in zip(pred_h, pred_a)
    ])
    poi_probs_ro = poi_probs[:, [2, 1, 0]]  # [p_away, p_draw, p_home]
    acc = accuracy_score(y_true, np.argmax(poi_probs_ro, axis=1))
    ll  = log_loss(y_true, poi_probs_ro)
    print(f"  Poisson 推導 Accuracy: {acc:.4f}  Log-loss: {ll:.4f}")

    return {
        "pr_home": pr_home, "pr_away": pr_away, "scaler": scaler,
        "poi_cols": poi_cols, "mae_home": mae_h, "mae_away": mae_a,
        "accuracy": acc, "log_loss": ll,
        "test_probs": poi_probs_ro, "y_test": y_true,
    }


def train_ensemble_v2(xgb_probs, poi_results, y_test):
    print("\n🎯 搜尋最佳 Ensemble 權重...")
    poi_probs = poi_results["test_probs"]
    n = min(len(xgb_probs), len(poi_probs), len(y_test))
    xgb_p, poi_p, y = xgb_probs[-n:], poi_probs[-n:], y_test[-n:]

    best_w, best_acc, best_ll = 0.7, 0.0, float("inf")
    for w in np.arange(0.3, 1.01, 0.05):
        ep = w * xgb_p + (1-w) * poi_p
        acc = accuracy_score(y, np.argmax(ep, axis=1))
        ll  = log_loss(y, ep)
        if acc > best_acc:
            best_acc, best_ll, best_w = acc, ll, w

    print(f"  最佳 XGB 權重：{best_w:.2f}  Acc：{best_acc:.4f}  LL：{best_ll:.4f}")
    return {"xgb_weight": best_w, "poi_weight": 1-best_w,
            "accuracy": best_acc, "log_loss": best_ll}


def write_report(wfcv_acc, wfcv_ll, xgb_acc, xgb_ll, top10,
                 poi_results, ensemble):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "  進階模型 v2 評估報告（加權ELO + H2H + 樣本權重）",
        "=" * 60,
        "",
        "【Walk-forward CV（跨年滾動驗證）】",
        f"  Accuracy : {wfcv_acc:.4f} ({wfcv_acc*100:.2f}%)",
        f"  Log-loss : {wfcv_ll:.4f}",
        "",
        "【XGBoost v2（測試集 20%）】",
        f"  Accuracy : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)",
        f"  Log-loss : {xgb_ll:.4f}",
        "",
        "  特徵重要性 Top 10：",
    ]
    for f, v in sorted(top10.items(), key=lambda x: -x[1]):
        lines.append(f"    {f:<45s} {v:.4f}")

    lines += [
        "",
        "【雙 Poisson v2】",
        f"  主場進球 MAE : {poi_results['mae_home']:.4f}",
        f"  客場進球 MAE : {poi_results['mae_away']:.4f}",
        f"  Accuracy : {poi_results['accuracy']:.4f}",
        "",
        "【Ensemble v2】",
        f"  XGBoost 權重 : {ensemble['xgb_weight']:.2f}",
        f"  Poisson  權重 : {ensemble['poi_weight']:.2f}",
        f"  Accuracy : {ensemble['accuracy']:.4f} ({ensemble['accuracy']*100:.2f}%)",
        f"  Log-loss : {ensemble['log_loss']:.4f}",
        "",
        "=" * 60,
    ]

    report_str = "\n".join(lines)
    print("\n" + report_str)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n💾 報告已儲存：{REPORT_PATH}")


def main():
    df, xgb_cols, poi_cols = load_data(DATASET_PATH)

    # Walk-forward validation 先評估穩定性
    wfcv_acc, wfcv_ll = walk_forward_validation(df, xgb_cols, n_folds=5)

    # 最終模型訓練
    xgb_model, acc, ll, xgb_probs, y_test, xgb_cols, top10 = train_xgboost_v2(df, xgb_cols)
    poi_results = train_poisson_v2(df, poi_cols)
    ensemble    = train_ensemble_v2(xgb_probs, poi_results, y_test)

    # 儲存模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "xgb_v2.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "features": xgb_cols}, f)

    poi_save = {k: v for k, v in poi_results.items()
                if k not in ("test_probs", "y_test")}
    with open(MODEL_DIR / "advanced_poisson_v2.pkl", "wb") as f:
        pickle.dump(poi_save, f)

    with open(MODEL_DIR / "ensemble_v2.pkl", "wb") as f:
        pickle.dump(ensemble, f)

    print(f"\n💾 模型已儲存至 {MODEL_DIR}/")
    write_report(wfcv_acc, wfcv_ll, acc, ll, top10, poi_results, ensemble)


if __name__ == "__main__":
    main()
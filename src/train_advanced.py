"""
train_advanced.py
=================
進階模型：XGBoost + 雙 Poisson + Ensemble

模型：
  1. XGBoost 分類器（勝/平/負）+ Optuna 超參數調優
  2. 雙 Poisson 回歸（主場進球、客場進球分別預測）
  3. Ensemble：XGBoost 勝/平/負機率 × Poisson 推導機率加權平均

輸入：
  data/processed/match_dataset.csv
  data/models/baseline_logistic.pkl  （用於 Ensemble 比較）

輸出：
  data/models/xgb_classifier.pkl
  data/models/advanced_poisson.pkl
  data/models/ensemble.pkl
  data/reports/advanced_report.txt

執行：
  pip install xgboost optuna scipy
  python src/train_advanced.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import xgboost as xgb
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, mean_absolute_error, brier_score_loss
)
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import poisson as scipy_poisson

warnings.filterwarnings("ignore")

# ── 路徑設定 ────────────────────────────────────────────────────
DATASET_PATH = r"data/processed/match_dataset.csv"
MODEL_DIR = Path(r"data/models")
REPORT_PATH = Path(r"data/reports/advanced_report.txt")
# ────────────────────────────────────────────────────────────────

# ── 特徵清單（比基準模型更完整）────────────────────────────────
XGB_FEATURES = [
    # ELO
    "home_elo", "away_elo", "elo_diff",
    # 環境
    "is_neutral", "is_wc_final", "is_friendly",
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
    "home_goals_for_avg_r5", "home_goals_against_avg_r5",
    "away_goals_for_avg_r5", "away_goals_against_avg_r5",
    "home_goal_diff_avg_r5", "away_goal_diff_avg_r5",
    "diff_goal_diff_avg_r5",
    "home_game_fw_finishing", "away_game_df_defense",
    "away_game_fw_finishing", "home_game_df_defense",
    "home_game_fw_speed", "away_game_fw_speed",
]
# ────────────────────────────────────────────────────────────────


def load_data(path):
    print("📥 讀取對陣樣本...")
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
            if g_h > g_a:
                p_home += p
            elif g_h == g_a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def train_xgboost(df, xgb_cols):
    print("\n🚀 訓練 XGBoost 分類器...")

    X = df[xgb_cols].values
    y = df["result"].values

    # 時序切分
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 類別比例（處理不平衡）
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  訓練集分布：{dict(zip(unique, counts))}")

    # ── Optuna 超參數調優 ──────────────────────────────────────
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma":            trial.suggest_float("gamma", 0, 5),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0, 2),
            }
            cv_losses = []
            for tr_idx, val_idx in tscv.split(X_train):
                xtr, xval = X_train[tr_idx], X_train[val_idx]
                ytr, yval = y_train[tr_idx], y_train[val_idx]
                clf = xgb.XGBClassifier(
                    **params,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=-1,
                )
                clf.fit(xtr, ytr, eval_set=[(xval, yval)], verbose=False)
                prob = clf.predict_proba(xval)
                cv_losses.append(log_loss(yval, prob))
            return np.mean(cv_losses)

        print("  🔍 Optuna 超參數搜索（50 trials）...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        best_params = study.best_params
        print(f"  最佳 log-loss: {study.best_value:.4f}")
        print(f"  最佳參數: {best_params}")

    except ImportError:
        print("  ⚠️  未安裝 optuna，使用預設參數")
        best_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    # ── 最終訓練 ──────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    print(f"\n  XGBoost Accuracy: {acc:.4f}  Log-loss: {ll:.4f}")

    # 特徵重要性
    feat_imp = pd.Series(xgb_model.feature_importances_, index=xgb_cols)
    top10 = feat_imp.nlargest(10)
    print("\n  特徵重要性 Top 10:")
    for f, v in top10.items():
        print(f"    {f:<45s} {v:.4f}")

    return xgb_model, acc, ll, y_prob, y_test, X_test, xgb_cols, top10.to_dict()


def train_dual_poisson(df, poi_cols):
    print("\n⚽ 訓練雙 Poisson 回歸（進球預測）...")

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

    # 搜尋最佳 alpha
    best_alpha_h = best_alpha_a = 1.0
    best_mae = float("inf")

    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        ph = PoissonRegressor(alpha=alpha, max_iter=500)
        pa = PoissonRegressor(alpha=alpha, max_iter=500)
        ph.fit(X_train_s, yh_train)
        pa.fit(X_train_s, ya_train)
        mae = (
            mean_absolute_error(yh_test, ph.predict(X_test_s)) +
            mean_absolute_error(ya_test, pa.predict(X_test_s))
        ) / 2
        if mae < best_mae:
            best_mae = mae
            best_alpha_h = best_alpha_a = alpha

    # 最終訓練
    pr_home = PoissonRegressor(alpha=best_alpha_h, max_iter=500)
    pr_away = PoissonRegressor(alpha=best_alpha_a, max_iter=500)
    pr_home.fit(X_train_s, yh_train)
    pr_away.fit(X_train_s, ya_train)

    pred_home = np.maximum(pr_home.predict(X_test_s), 0.05)
    pred_away = np.maximum(pr_away.predict(X_test_s), 0.05)

    mae_h = mean_absolute_error(yh_test, pred_home)
    mae_a = mean_absolute_error(ya_test, pred_away)
    print(f"  主場進球 MAE: {mae_h:.4f}  客場進球 MAE: {mae_a:.4f}")

    # 轉換為勝/平/負機率
    y_true = df["result"].values[split:]
    poi_probs = np.array([
        poisson_match_probs(lh, la)
        for lh, la in zip(pred_home, pred_away)
    ])  # shape (N, 3)：[p_home, p_draw, p_away]

    # 重排為 [p_away, p_draw, p_home]（配合 result: 0=客勝,1=平,2=主勝）
    poi_probs_reordered = poi_probs[:, [2, 1, 0]]
    y_pred_poi = np.argmax(poi_probs_reordered, axis=1)
    acc_poi = accuracy_score(y_true, y_pred_poi)
    ll_poi = log_loss(y_true, poi_probs_reordered)
    print(f"  Poisson 推導 Accuracy: {acc_poi:.4f}  Log-loss: {ll_poi:.4f}")

    return {
        "pr_home": pr_home,
        "pr_away": pr_away,
        "scaler": scaler,
        "poi_cols": poi_cols,
        "mae_home": mae_h,
        "mae_away": mae_a,
        "accuracy": acc_poi,
        "log_loss": ll_poi,
        "test_probs": poi_probs_reordered,
        "y_test": y_true,
    }


def train_ensemble(xgb_probs, poi_results, y_test):
    """
    Ensemble：搜尋 XGBoost 與 Poisson 的最佳加權比例
    """
    print("\n🎯 訓練 Ensemble（XGBoost + Poisson 加權）...")

    poi_probs = poi_results["test_probs"]

    # 保證兩者長度一致
    n = min(len(xgb_probs), len(poi_probs), len(y_test))
    xgb_p = xgb_probs[-n:]
    poi_p = poi_probs[-n:]
    y = y_test[-n:]

    best_w = 0.7
    best_acc = 0.0
    best_ll = float("inf")

    for w in np.arange(0.3, 1.01, 0.05):
        ensemble_p = w * xgb_p + (1 - w) * poi_p
        y_pred = np.argmax(ensemble_p, axis=1)
        acc = accuracy_score(y, y_pred)
        ll = log_loss(y, ensemble_p)
        if acc > best_acc:
            best_acc = acc
            best_ll = ll
            best_w = w

    print(f"  最佳 XGBoost 權重：{best_w:.2f}  Accuracy：{best_acc:.4f}  Log-loss：{best_ll:.4f}")

    return {
        "xgb_weight": best_w,
        "poi_weight": 1 - best_w,
        "accuracy": best_acc,
        "log_loss": best_ll,
    }


def save_all(xgb_model, xgb_cols, poi_results, ensemble_params):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_DIR / "xgb_classifier.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "features": xgb_cols}, f)

    poi_save = {k: v for k, v in poi_results.items()
                if k not in ("test_probs", "y_test")}
    with open(MODEL_DIR / "advanced_poisson.pkl", "wb") as f:
        pickle.dump(poi_save, f)

    with open(MODEL_DIR / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble_params, f)

    print(f"\n💾 模型已儲存至 {MODEL_DIR}/")


def write_report(xgb_acc, xgb_ll, xgb_top_feats, poi_results, ensemble):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "  進階模型評估報告",
        "=" * 60,
        "",
        "【XGBoost 分類器（勝/平/負）】",
        f"  Accuracy : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)",
        f"  Log-loss : {xgb_ll:.4f}",
        "",
        "  特徵重要性 Top 10：",
    ]
    for feat, imp in sorted(xgb_top_feats.items(), key=lambda x: -x[1]):
        lines.append(f"    {feat:<45s} {imp:.4f}")

    lines += [
        "",
        "【雙 Poisson 進球預測】",
        f"  主場進球 MAE : {poi_results['mae_home']:.4f}",
        f"  客場進球 MAE : {poi_results['mae_away']:.4f}",
        f"  推導勝/平/負 Accuracy : {poi_results['accuracy']:.4f}",
        f"  Log-loss : {poi_results['log_loss']:.4f}",
        "",
        "【Ensemble 整合】",
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

    xgb_model, acc, ll, xgb_probs, y_test, X_test, xgb_cols, top10 = train_xgboost(df, xgb_cols)
    poi_results = train_dual_poisson(df, poi_cols)
    ensemble = train_ensemble(xgb_probs, poi_results, y_test)

    save_all(xgb_model, xgb_cols, poi_results, ensemble)
    write_report(acc, ll, top10, poi_results, ensemble)


if __name__ == "__main__":
    main()

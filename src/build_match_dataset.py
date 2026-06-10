"""
build_match_dataset.py
======================
從歷史賽事 + 國家隊特徵表，建立模型訓練用的「對陣樣本」。

每一行 = 一場比賽，特徵包含：
  - ELO 分數（動態，反映當場開賽前的強弱）
  - 主客隊各項特徵（身價、EA OVR、歷史勝率、近期狀態等）
  - 主客隊差值特徵（最直接的強弱對比）
  - 比賽環境（中立場、賽事類型）

輸入：
  data/processed/cleaned_results.csv
  data/processed/wc26_final_features.csv

輸出：
  data/processed/match_dataset.csv     完整對陣樣本（49,000+ 筆）
  data/processed/wc26_elo.csv          48 隊最終 ELO 分數

執行：
  python src/build_match_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑設定 ────────────────────────────────────────────────────
RESULTS_PATH = r"data/processed/cleaned_results.csv"
FEATURES_PATH = r"data/processed/wc26_final_features.csv"
OUT_DATASET = r"data/processed/match_dataset.csv"
OUT_ELO = r"data/processed/wc26_elo.csv"
# ────────────────────────────────────────────────────────────────

# ELO 參數
ELO_DEFAULT = 1500
ELO_K = 32


def compute_elo(df: pd.DataFrame, k: int = ELO_K, default: int = ELO_DEFAULT):
    """
    對每場比賽記錄「開賽前」的 ELO，並更新到下一場。
    回傳加上 home_elo / away_elo / elo_diff 欄位的 df，
    以及最終各隊 ELO 字典。
    """
    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        he = elo.get(h, default)
        ae = elo.get(a, default)
        home_elos.append(he)
        away_elos.append(ae)

        # 期望值
        we_h = 1 / (1 + 10 ** ((ae - he) / 400))
        we_a = 1 - we_h

        # 實際結果
        if row["home_result"] == "W":
            wh, wa = 1.0, 0.0
        elif row["home_result"] == "D":
            wh, wa = 0.5, 0.5
        else:
            wh, wa = 0.0, 1.0

        # 更新
        elo[h] = he + k * (wh - we_h)
        elo[a] = ae + k * (wa - we_a)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df, elo


def build_match_dataset():
    print("📥 讀取資料...")
    results = pd.read_csv(RESULTS_PATH, low_memory=False)
    feat = pd.read_csv(FEATURES_PATH)

    # ── 只取歷史賽事（排除未來的 2026 WC 分組賽）──
    hist = results[results["is_future"] != True].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    print(f"  歷史賽事：{len(hist):,} 筆")
    print(f"  國家隊特徵：{len(feat)} 隊 × {len(feat.columns)} 欄")

    # ── ELO 計算（全時期，保留每場開賽前的 ELO）──
    print("\n🔢 計算 ELO 分數...")
    hist, final_elo = compute_elo(hist)

    # 儲存 48 隊最終 ELO
    wc_teams = feat["team"].tolist()
    elo_df = pd.DataFrame([
        {"team": t, "elo": round(final_elo.get(t, ELO_DEFAULT), 2)}
        for t in wc_teams
    ]).sort_values("elo", ascending=False)
    Path(OUT_ELO).parent.mkdir(parents=True, exist_ok=True)
    elo_df.to_csv(OUT_ELO, index=False, encoding="utf-8-sig")
    print(f"  ELO Top5: {elo_df.head(5)[['team','elo']].values.tolist()}")

    # ── 特徵查找表（帶缺值填補）──
    feat_cols = [c for c in feat.columns if c != "team"]
    feat_indexed = feat.set_index("team")
    feat_means = feat[feat_cols].mean()  # 全域均值，用於找不到特徵的隊伍

    def get_feat(team, col):
        if team in feat_indexed.index:
            val = feat_indexed.at[team, col]
            return val if pd.notna(val) else feat_means[col]
        return feat_means[col]

    # ── 建立對陣樣本 ──
    print("\n🔧 建立對陣樣本...")
    records = []
    for _, row in hist.iterrows():
        h, a = row["home_team"], row["away_team"]

        rec = {
            # 識別資訊
            "date":             row["date"],
            "home_team":        h,
            "away_team":        a,
            "tournament_type":  row["tournament_type"],
            # 環境特徵
            "is_neutral":       int(row["neutral"] == True),
            "is_wc_final":      int(row["tournament_type"] == "WC_final"),
            "is_friendly":      int(row["tournament_type"] == "Friendly"),
            # ELO
            "home_elo":         row["home_elo"],
            "away_elo":         row["away_elo"],
            "elo_diff":         row["elo_diff"],
            # 標籤
            "result":           {"W": 2, "D": 1, "L": 0}[row["home_result"]],
            "home_score":       row["home_score"],
            "away_score":       row["away_score"],
        }

        # 主客隊各項特徵
        for col in feat_cols:
            rec[f"home_{col}"] = get_feat(h, col)
            rec[f"away_{col}"] = get_feat(a, col)

        records.append(rec)

    df = pd.DataFrame(records)

    # ── 差值特徵（主隊 - 客隊）──
    diff_targets = [
        "total_value_eur",
        "game_top_11_ovr",
        "game_fw_speed",
        "game_fw_finishing",
        "game_mf_passing",
        "game_df_defense",
        "game_df_physic",
        "win_rate_r5",
        "win_rate_wc",
        "goal_diff_avg_r5",
        "goals_for_avg_r5",
        "goals_against_avg_r5",
        "form_score_official",
        "top_club_ratio",
        "avg_age",
    ]
    for col in diff_targets:
        if f"home_{col}" in df.columns:
            df[f"diff_{col}"] = df[f"home_{col}"] - df[f"away_{col}"]

    # 身價差轉成百萬歐元（避免數值過大）
    df["diff_value_M"] = df["diff_total_value_eur"] / 1e6
    df = df.drop(columns=["diff_total_value_eur"], errors="ignore")

    # ── 缺值確認 ──
    null_count = df.isnull().sum().sum()
    print(f"  空值數量：{null_count}（應為 0）")

    # ── 儲存 ──
    df.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")

    print(f"\n💾 儲存：{OUT_DATASET}")
    print(f"   {len(df):,} 筆 × {len(df.columns)} 欄")
    print(f"\n📊 result 分布：")
    dist = df["result"].value_counts().sort_index()
    labels = {0: "客勝(0)", 1: "平局(1)", 2: "主勝(2)"}
    for k, v in dist.items():
        print(f"  {labels[k]}: {v:,} ({v/len(df)*100:.1f}%)")


if __name__ == "__main__":
    build_match_dataset()

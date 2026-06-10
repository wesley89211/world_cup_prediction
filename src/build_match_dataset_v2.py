"""
build_match_dataset_v2.py
==========================
優化版對陣樣本產生器，包含三項改進：

改進一：加權 ELO（賽事重要性）
  - WC 決賽圈 K×2.0、洲際賽 K×1.5、WC 外圍賽 K×1.25、友誼賽 K×0.5
  - 中立場主場優勢修正（ELO 差距縮小 50 分）

改進二：H2H 歷史交手特徵
  - h2h_total（交手次數）
  - h2h_home_winrate（主隊對客隊歷史勝率）
  - h2h_draw_rate（歷史平局率）
  - 非 WC 隊對陣填入全局均值 (0.333)

改進三：樣本權重（訓練時使用）
  - 時間衰減：decay=0.999/天，近期比賽影響力更大
  - 賽事加權：WC × 2.0，友誼賽 × 0.5
  - 最終存入 sample_weight 欄位，供 XGBoost fit() 使用

輸入：
  data/processed/cleaned_results.csv
  data/processed/wc26_final_features.csv
  data/processed/wc26_h2h.csv

輸出：
  data/processed/match_dataset_v2.csv    優化版對陣樣本
  data/processed/wc26_elo_v2.csv         加權 ELO 最終分數

執行：
  python src/build_match_dataset_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑設定 ─────────────────────────────────────────────────────
RESULTS_PATH  = r"data/processed/cleaned_results.csv"
FEATURES_PATH = r"data/processed/wc26_final_features.csv"
H2H_PATH      = r"data/processed/wc26_h2h.csv"
OUT_DATASET   = r"data/processed/match_dataset_v2.csv"
OUT_ELO       = r"data/processed/wc26_elo_v2.csv"
# ─────────────────────────────────────────────────────────────────

ELO_DEFAULT = 1500
ELO_K_BASE  = 32

# 賽事重要性 K 值乘數
TOURNAMENT_K = {
    "WC_final":          2.00,
    "Major_Continental": 1.50,
    "WC_qual":           1.25,
    "Other":             1.00,
    "Friendly":          0.50,
}

# 樣本權重：賽事加權
TOURNAMENT_W = {
    "WC_final":          2.00,
    "Major_Continental": 1.50,
    "WC_qual":           1.25,
    "Other":             1.00,
    "Friendly":          0.50,
}

# 時間衰減（每天 0.1% 衰減）
TIME_DECAY = 0.999

# H2H 預設值（雙方未曾交手時）
H2H_DEFAULT_WINRATE  = 0.333
H2H_DEFAULT_DRAWRATE = 0.250


# ════════════════════════════════════════════════════════════════
# 改進一：加權 ELO
# ════════════════════════════════════════════════════════════════

def compute_weighted_elo(df: pd.DataFrame):
    """
    對每場比賽記錄「開賽前」的加權 ELO。

    改進點：
    1. K 值依賽事重要性調整（WC 決賽圈 K=64，友誼賽 K=16）
    2. 中立場主場優勢修正：標準 ELO 假設主場有優勢，
       中立場則無，修正方式是將主隊 ELO 暫時減少 HOME_ADV 分
    """
    HOME_ADV = 50  # 主場優勢修正值（ELO 分）

    elo: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        he_raw = elo.get(h, ELO_DEFAULT)
        ae_raw = elo.get(a, ELO_DEFAULT)

        # 主場優勢修正：非中立場，主隊加 HOME_ADV
        neutral = row.get("neutral", False)
        if str(neutral).lower() in ("true", "1", "yes"):
            he_adj = he_raw
        else:
            he_adj = he_raw + HOME_ADV

        home_elos.append(he_raw)   # 記錄原始 ELO（不含臨時修正）
        away_elos.append(ae_raw)

        # 期望值
        we_h = 1 / (1 + 10 ** ((ae_raw - he_adj) / 400))
        we_a = 1 - we_h

        # 實際結果
        if row["home_result"] == "W":
            wh, wa = 1.0, 0.0
        elif row["home_result"] == "D":
            wh, wa = 0.5, 0.5
        else:
            wh, wa = 0.0, 1.0

        # 加權 K
        k = ELO_K_BASE * TOURNAMENT_K.get(row.get("tournament_type", "Other"), 1.0)

        # 更新
        elo[h] = he_raw + k * (wh - we_h)
        elo[a] = ae_raw + k * (wa - we_a)

    df = df.copy()
    df["home_elo"]  = home_elos
    df["away_elo"]  = away_elos
    df["elo_diff"]  = [h - a for h, a in zip(home_elos, away_elos)]
    return df, elo


# ════════════════════════════════════════════════════════════════
# 改進二：H2H 特徵
# ════════════════════════════════════════════════════════════════

def build_h2h_lookup(h2h_df: pd.DataFrame) -> dict:
    """
    建立雙向查找字典。
    key = (home_team, away_team)
    value = {h2h_total, h2h_home_winrate, h2h_draw_rate}
    """
    lookup = {}
    for _, r in h2h_df.iterrows():
        a, b = r["team_a"], r["team_b"]
        total = r["total_matches"]
        if total == 0:
            continue

        # team_a 視角
        lookup[(a, b)] = {
            "h2h_total":        int(total),
            "h2h_home_winrate": round(r["team_a_wins"] / total, 4),
            "h2h_draw_rate":    round(r["draws"] / total, 4),
        }
        # team_b 視角（home/away 互換）
        lookup[(b, a)] = {
            "h2h_total":        int(total),
            "h2h_home_winrate": round(r["team_b_wins"] / total, 4),
            "h2h_draw_rate":    round(r["draws"] / total, 4),
        }
    return lookup


def get_h2h(home: str, away: str, lookup: dict) -> dict:
    """查詢 H2H 特徵，找不到時填預設值"""
    if (home, away) in lookup:
        return lookup[(home, away)]
    return {
        "h2h_total":        0,
        "h2h_home_winrate": H2H_DEFAULT_WINRATE,
        "h2h_draw_rate":    H2H_DEFAULT_DRAWRATE,
    }


# ════════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════════

def build_match_dataset_v2():
    print("📥 讀取資料...")
    results  = pd.read_csv(RESULTS_PATH, low_memory=False)
    feat     = pd.read_csv(FEATURES_PATH)
    h2h_raw  = pd.read_csv(H2H_PATH)

    hist = results[results["is_future"] != True].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    print(f"  歷史賽事：{len(hist):,} 筆")
    print(f"  國家隊特徵：{len(feat)} 隊 × {len(feat.columns)} 欄")
    print(f"  H2H 交手組合：{len(h2h_raw)} 組")

    # ── 加權 ELO ──
    print("\n🔢 計算加權 ELO（賽事重要性 + 主場優勢修正）...")
    hist, final_elo = compute_weighted_elo(hist)

    # 儲存 48 隊最終 ELO
    wc_teams = feat["team"].tolist()
    elo_df = pd.DataFrame([
        {"team": t, "elo_weighted": round(final_elo.get(t, ELO_DEFAULT), 2)}
        for t in wc_teams
    ]).sort_values("elo_weighted", ascending=False)
    Path(OUT_ELO).parent.mkdir(parents=True, exist_ok=True)
    elo_df.to_csv(OUT_ELO, index=False, encoding="utf-8-sig")
    print(f"  ELO Top5: {elo_df.head(5)[['team','elo_weighted']].values.tolist()}")

    # ── H2H lookup ──
    print("\n🤝 建立 H2H 查找表...")
    h2h_lookup = build_h2h_lookup(h2h_raw)
    print(f"  H2H 組合數：{len(h2h_lookup)} 筆（雙向）")

    # ── 特徵查找表 ──
    feat_cols    = [c for c in feat.columns if c != "team"]
    feat_indexed = feat.set_index("team")
    feat_means   = feat[feat_cols].mean()

    def get_feat(team, col):
        if team in feat_indexed.index:
            val = feat_indexed.at[team, col]
            return val if pd.notna(val) else feat_means[col]
        return feat_means[col]

    # ── 改進三：樣本權重 ──
    max_date = hist["date"].max()
    hist["days_ago"] = (max_date - hist["date"]).dt.days
    hist["time_weight"] = TIME_DECAY ** hist["days_ago"]
    hist["tournament_weight"] = hist["tournament_type"].map(TOURNAMENT_W).fillna(1.0)
    hist["sample_weight"] = (hist["time_weight"] * hist["tournament_weight"]).round(6)

    # ── 建立對陣樣本 ──
    print("\n🔧 建立對陣樣本（含 H2H + 樣本權重）...")
    records = []

    for _, row in hist.iterrows():
        h, a = row["home_team"], row["away_team"]

        h2h_feat = get_h2h(h, a, h2h_lookup)

        rec = {
            # 識別資訊
            "date":            row["date"],
            "home_team":       h,
            "away_team":       a,
            "tournament_type": row["tournament_type"],
            # 環境特徵
            "is_neutral":      int(str(row.get("neutral","False")).lower() in ("true","1")),
            "is_wc_final":     int(row["tournament_type"] == "WC_final"),
            "is_friendly":     int(row["tournament_type"] == "Friendly"),
            # 加權 ELO
            "home_elo":        row["home_elo"],
            "away_elo":        row["away_elo"],
            "elo_diff":        row["elo_diff"],
            # H2H 特徵
            "h2h_total":           h2h_feat["h2h_total"],
            "h2h_home_winrate":    h2h_feat["h2h_home_winrate"],
            "h2h_draw_rate":       h2h_feat["h2h_draw_rate"],
            # 樣本權重
            "sample_weight":   row["sample_weight"],
            # 標籤
            "result":          {"W": 2, "D": 1, "L": 0}[row["home_result"]],
            "home_score":      row["home_score"],
            "away_score":      row["away_score"],
        }

        # 主客隊各項特徵
        for col in feat_cols:
            rec[f"home_{col}"] = get_feat(h, col)
            rec[f"away_{col}"] = get_feat(a, col)

        records.append(rec)

    df = pd.DataFrame(records)

    # ── 差值特徵 ──
    diff_targets = [
        "total_value_eur", "game_top_11_ovr", "game_fw_speed",
        "game_fw_finishing", "game_mf_passing", "game_df_defense",
        "game_df_physic", "win_rate_r5", "win_rate_wc",
        "goal_diff_avg_r5", "goals_for_avg_r5", "goals_against_avg_r5",
        "form_score_official", "top_club_ratio", "avg_age",
    ]
    for col in diff_targets:
        if f"home_{col}" in df.columns:
            df[f"diff_{col}"] = df[f"home_{col}"] - df[f"away_{col}"]

    df["diff_value_M"] = df["diff_total_value_eur"] / 1e6
    df = df.drop(columns=["diff_total_value_eur"], errors="ignore")

    # ── 缺值確認 ──
    null_count = df.isnull().sum().sum()
    print(f"  空值數量：{null_count}（應為 0）")

    # ── 儲存 ──
    df.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")

    print(f"\n💾 儲存：{OUT_DATASET}")
    print(f"   {len(df):,} 筆 × {len(df.columns)} 欄")

    print(f"\n📊 新增特徵確認：")
    new_feats = ["h2h_total","h2h_home_winrate","h2h_draw_rate","sample_weight"]
    for f in new_feats:
        print(f"  {f}: mean={df[f].mean():.4f}  null={df[f].isna().sum()}")

    print(f"\n📊 result 分布：")
    dist = df["result"].value_counts().sort_index()
    labels = {0:"客勝(0)", 1:"平局(1)", 2:"主勝(2)"}
    for k, v in dist.items():
        print(f"  {labels[k]}: {v:,} ({v/len(df)*100:.1f}%)")

    print(f"\n📊 樣本權重分布（賽事 × 時間衰減）：")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.1]
    labels_w = ["0-0.1","0.1-0.3","0.3-0.5","0.5-0.7","0.7-1.0","1.0-2.0"]
    counts = pd.cut(df["sample_weight"], bins=bins, labels=labels_w).value_counts().sort_index()
    for label, count in counts.items():
        print(f"  weight {label}: {count:,} 筆")


if __name__ == "__main__":
    build_match_dataset_v2()

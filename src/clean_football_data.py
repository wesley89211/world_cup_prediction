"""
2026 FIFA World Cup — 歷史對戰資料清整腳本
==========================================
輸入檔案（放在同一目錄下）：
  results.csv
  goalscorers.csv
  shootouts.csv
  former_names.csv
  wc2026_squads_final.csv   ← 上一步產出的大名單

輸出檔案：
  cleaned_results.csv       完整清整後的歷史賽事
  wc26_team_stats.csv       48支世界盃隊伍的統計特徵
  wc26_h2h.csv              世界盃隊伍之間的兩兩交手紀錄
  wc26_form.csv             各隊近期狀態分（最近10場加權）

執行：
  pip install pandas numpy
  python clean_football_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑設定（改成你的實際路徑）──────────────────────────────────
DATA_DIR = Path(".")   # 所有 CSV 放在同一層
OUT_DIR  = Path(".")   # 輸出也放同一層
# ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# 1. 讀取原始資料
# ════════════════════════════════════════════════════════════════
print("📥 讀取原始資料...")

results     = pd.read_csv(DATA_DIR / "results.csv")
goalscorers = pd.read_csv(DATA_DIR / "goalscorers.csv")
shootouts   = pd.read_csv(DATA_DIR / "shootouts.csv")
former      = pd.read_csv(DATA_DIR / "former_names.csv")
squads      = pd.read_csv(DATA_DIR / "wc2026_squads_final.csv")

print(f"  results:     {len(results):,} 筆")
print(f"  goalscorers: {len(goalscorers):,} 筆")
print(f"  shootouts:   {len(shootouts):,} 筆")


# ════════════════════════════════════════════════════════════════
# 2. 標準化隊名：舊名 → 現用名
#    former_names.csv 記錄了哪些隊名在特定日期前用舊名
# ════════════════════════════════════════════════════════════════
print("\n🔧 標準化隊名...")

def build_name_mapper(former_df):
    """
    回傳 {舊名: 現用名} dict。
    注意：只把無歧義的舊名映射過去（同一 former 只對應一個 current）。
    """
    mapper = {}
    for _, row in former_df.iterrows():
        mapper[row["former"]] = row["current"]
    return mapper

name_map = build_name_mapper(former)

def normalize_team(df, cols):
    """把指定欄位的隊名全部換成現用名。"""
    for col in cols:
        df[col] = df[col].replace(name_map)
    return df

results     = normalize_team(results,     ["home_team", "away_team"])
goalscorers = normalize_team(goalscorers, ["home_team", "away_team", "team"])
shootouts   = normalize_team(shootouts,   ["home_team", "away_team", "winner"])

print(f"  映射規則數：{len(name_map)}")


# ════════════════════════════════════════════════════════════════
# 3. 日期型別轉換
# ════════════════════════════════════════════════════════════════
print("\n📅 轉換日期格式...")

for df in [results, goalscorers, shootouts, former]:
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

results["year"] = results["date"].dt.year


# ════════════════════════════════════════════════════════════════
# 4. 處理 score 缺值
#    72 筆 null 全是 2026 世界盃「尚未開打」的賽程，
#    保留但加上 is_future 旗標，不納入歷史統計。
# ════════════════════════════════════════════════════════════════
print("\n🔍 處理缺值...")

results["is_future"] = results["home_score"].isna()

# 歷史賽事：score 完整
hist = results[~results["is_future"]].copy()
future = results[results["is_future"]].copy()

print(f"  歷史賽事：{len(hist):,} 筆")
print(f"  未來賽程：{len(future):,} 筆（2026 WC 分組賽，score=NaN）")


# ════════════════════════════════════════════════════════════════
# 5. 加入延伸欄位
# ════════════════════════════════════════════════════════════════
print("\n➕ 計算延伸欄位...")

hist = hist.copy()

# 比賽結果（從主場角度）
hist["home_result"] = np.where(
    hist["home_score"] > hist["away_score"], "W",
    np.where(hist["home_score"] < hist["away_score"], "L", "D")
)
hist["total_goals"] = hist["home_score"] + hist["away_score"]

# 加入是否有PK
shootout_keys = shootouts[["date", "home_team", "away_team", "winner"]].copy()
hist = hist.merge(shootout_keys, on=["date", "home_team", "away_team"], how="left")
hist["had_shootout"] = hist["winner"].notna()
hist = hist.rename(columns={"winner": "shootout_winner"})

# 賽事類型簡化分類
def classify_tournament(t):
    t_lower = t.lower()
    if "world cup" in t_lower and "qualif" not in t_lower:
        return "WC_final"
    elif "world cup qualif" in t_lower:
        return "WC_qual"
    elif "friendly" in t_lower:
        return "Friendly"
    elif any(x in t_lower for x in ["euro", "copa", "nations league", "gold cup",
                                     "african cup", "asian cup", "concacaf"]):
        return "Major_Continental"
    else:
        return "Other"

hist["tournament_type"] = hist["tournament"].apply(classify_tournament)
print("  賽事類型分布：")
print(hist["tournament_type"].value_counts().to_string())


# ════════════════════════════════════════════════════════════════
# 6. 儲存完整清整結果
# ════════════════════════════════════════════════════════════════
print("\n💾 儲存 cleaned_results.csv ...")

all_cleaned = pd.concat([hist, future], ignore_index=True).sort_values("date")
all_cleaned.to_csv(OUT_DIR / "cleaned_results.csv", index=False, encoding="utf-8-sig")
print(f"  共 {len(all_cleaned):,} 筆")


# ════════════════════════════════════════════════════════════════
# 7. 計算 48 隊的統計特徵
#    只計算「歷史賽事」，並提供三個視角：
#      (a) 全時期
#      (b) 近 5 年（2021–）
#      (c) 僅世界盃決賽圈
# ════════════════════════════════════════════════════════════════
print("\n📊 計算各隊統計特徵...")

WC_TEAMS = set(squads["Country"].unique())
RECENT_CUTOFF = pd.Timestamp("2021-01-01")

def team_stats(df, teams, label="all"):
    """
    把 home/away 各自展開為「該隊視角」，計算勝率、進失球等。
    """
    records = []
    for team in teams:
        home = df[df["home_team"] == team].copy()
        away = df[df["away_team"] == team].copy()

        # 統一視角：goals_for, goals_against, result
        home["gf"] = home["home_score"]
        home["ga"] = home["away_score"]
        home["result"] = home["home_result"]

        away["gf"] = away["away_score"]
        away["ga"] = away["home_score"]
        away["result"] = away["home_result"].map({"W": "L", "L": "W", "D": "D"})

        combined = pd.concat([home[["date","gf","ga","result","tournament_type","neutral"]],
                              away[["date","gf","ga","result","tournament_type","neutral"]]])

        n = len(combined)
        if n == 0:
            continue

        w = (combined["result"] == "W").sum()
        d = (combined["result"] == "D").sum()
        l = (combined["result"] == "L").sum()

        records.append({
            "team": team,
            "period": label,
            "matches": n,
            "wins": w,
            "draws": d,
            "losses": l,
            "win_rate": round(w / n, 4),
            "goals_for_avg": round(combined["gf"].mean(), 3),
            "goals_against_avg": round(combined["ga"].mean(), 3),
            "goal_diff_avg": round((combined["gf"] - combined["ga"]).mean(), 3),
        })

    return pd.DataFrame(records)

stats_all    = team_stats(hist, WC_TEAMS, label="all")
stats_recent = team_stats(hist[hist["date"] >= RECENT_CUTOFF], WC_TEAMS, label="recent_5yr")
stats_wc     = team_stats(hist[hist["tournament_type"] == "WC_final"], WC_TEAMS, label="WC_finals_only")

team_stats_df = pd.concat([stats_all, stats_recent, stats_wc], ignore_index=True)
team_stats_df.to_csv(OUT_DIR / "wc26_team_stats.csv", index=False, encoding="utf-8-sig")
print(f"  wc26_team_stats.csv：{len(team_stats_df)} 筆（{len(WC_TEAMS)} 隊 × 3 期間）")


# ════════════════════════════════════════════════════════════════
# 8. 兩隊交手紀錄（H2H）
#    只計算 48 隊之間的對戰，方便分組賽查詢
# ════════════════════════════════════════════════════════════════
print("\n⚔️  計算兩隊交手紀錄...")

wc_hist = hist[
    hist["home_team"].isin(WC_TEAMS) & hist["away_team"].isin(WC_TEAMS)
].copy()

h2h_records = []
teams_list = sorted(WC_TEAMS)

for i, team_a in enumerate(teams_list):
    for team_b in teams_list[i+1:]:
        matches = wc_hist[
            ((wc_hist["home_team"] == team_a) & (wc_hist["away_team"] == team_b)) |
            ((wc_hist["home_team"] == team_b) & (wc_hist["away_team"] == team_a))
        ]

        if len(matches) == 0:
            continue

        # team_a 視角
        a_wins = d = b_wins = 0
        for _, m in matches.iterrows():
            if m["home_team"] == team_a:
                gf, ga = m["home_score"], m["away_score"]
            else:
                gf, ga = m["away_score"], m["home_score"]

            if gf > ga:   a_wins += 1
            elif gf < ga: b_wins += 1
            else:         d += 1

        last_match = matches.sort_values("date").iloc[-1]
        last_winner = (
            last_match["home_team"] if last_match["home_result"] == "W"
            else last_match["away_team"] if last_match["home_result"] == "L"
            else "Draw"
        )

        h2h_records.append({
            "team_a": team_a,
            "team_b": team_b,
            "total_matches": len(matches),
            "team_a_wins": a_wins,
            "draws": d,
            "team_b_wins": b_wins,
            "last_match_date": last_match["date"].date(),
            "last_match_winner": last_winner,
            "last_score": f"{int(last_match['home_score'])}-{int(last_match['away_score'])}",
            "last_home_team": last_match["home_team"],
        })

h2h_df = pd.DataFrame(h2h_records)
h2h_df.to_csv(OUT_DIR / "wc26_h2h.csv", index=False, encoding="utf-8-sig")
print(f"  wc26_h2h.csv：{len(h2h_df):,} 組交手紀錄")


# ════════════════════════════════════════════════════════════════
# 9. 近期狀態分（Form Score）
#    最近 10 場：勝=3分、平=1分、負=0分
#    加權：越近的比賽權重越高（指數衰減）
# ════════════════════════════════════════════════════════════════
print("\n📈 計算近期狀態分...")

def compute_form(df, team, n=10, decay=0.85):
    """
    取最近 n 場，用指數衰減加權計算 form score。
    decay：越小代表越重視最近的比賽（0.85 = 合理值）
    """
    home = df[df["home_team"] == team][["date","home_result"]].rename(
        columns={"home_result": "result"})
    away = df[df["away_team"] == team][["date","home_result"]].copy()
    away["result"] = away["home_result"].map({"W": "L", "L": "W", "D": "D"})
    away = away[["date","result"]]

    combined = pd.concat([home, away]).sort_values("date", ascending=False).head(n)

    if len(combined) == 0:
        return 0.0

    points = combined["result"].map({"W": 3, "D": 1, "L": 0}).values
    weights = np.array([decay**i for i in range(len(points))])
    score = np.dot(points, weights) / weights.sum() / 3  # 正規化到 0~1

    return round(score, 4)

form_records = []
for team in WC_TEAMS:
    # 全類型比賽
    form_all = compute_form(hist, team, n=10)
    # 僅正式賽事（非友誼賽）
    form_official = compute_form(
        hist[hist["tournament_type"] != "Friendly"], team, n=10)

    # 最近10場詳情
    home = hist[hist["home_team"] == team][["date","away_team","home_result","tournament_type"]].rename(
        columns={"away_team":"opponent","home_result":"result"})
    away = hist[hist["away_team"] == team][["date","home_team","home_result","tournament_type"]].rename(
        columns={"home_team":"opponent","home_result":"result"})
    away["result"] = away["result"].map({"W":"L","L":"W","D":"D"})

    last10 = pd.concat([home,away]).sort_values("date",ascending=False).head(10)
    last10_str = " ".join(last10["result"].values)  # e.g. "W W D L W W W D W L"

    form_records.append({
        "team": team,
        "form_score_all": form_all,
        "form_score_official": form_official,
        "last_10_results": last10_str,
        "last_match_date": last10["date"].iloc[0].date() if len(last10) > 0 else None,
    })

form_df = pd.DataFrame(form_records).sort_values("form_score_official", ascending=False)
form_df.to_csv(OUT_DIR / "wc26_form.csv", index=False, encoding="utf-8-sig")
print(f"  wc26_form.csv：{len(form_df)} 隊")


# ════════════════════════════════════════════════════════════════
# 10. 完成報告
# ════════════════════════════════════════════════════════════════
print("\n" + "="*52)
print("✅ 清整完成！輸出檔案：")
print("  cleaned_results.csv   — 完整歷史賽事（含未來賽程旗標）")
print("  wc26_team_stats.csv   — 各隊統計（全期/近5年/WC決賽圈）")
print("  wc26_h2h.csv          — 48隊兩兩交手紀錄")
print("  wc26_form.csv         — 近期狀態分")
print()
print("📌 使用建議：")
print("  預測模型特徵推薦組合：")
print("    win_rate (recent_5yr) + goal_diff_avg (recent_5yr)")
print("    + form_score_official + WC_finals win_rate")
print("    → 加上大名單的身價/OVR → Poisson / Elo 模型")

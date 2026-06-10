"""
build_final_features.py
========================
將歷史賽事特徵（team_stats / form）合併進 wc26_squad_features.csv
產出最終特徵表，可直接用於預測模型。

輸入：
  data/processed/wc26_squad_features.csv   EA + 身價特徵（48隊 × 13欄）
  data/processed/wc26_team_stats.csv       歷史勝率（3個時期）
  data/processed/wc26_form.csv             近期狀態分

輸出：
  data/processed/wc26_final_features.csv   完整特徵表（48隊 × 26欄）

執行：
  python src/build_final_features.py
"""

import pandas as pd
import numpy as np

# ── 路徑設定 ────────────────────────────────────────────────────
SQUAD_FEAT_PATH = r"data/processed/wc26_squad_features.csv"
TEAM_STATS_PATH = r"data/processed/wc26_team_stats.csv"
FORM_PATH       = r"data/processed/wc26_form.csv"
OUT_PATH        = r"data/processed/wc26_final_features.csv"
# ────────────────────────────────────────────────────────────────


def build_final_features():
    print("📥 讀取資料...")
    squad = pd.read_csv(SQUAD_FEAT_PATH)
    stats = pd.read_csv(TEAM_STATS_PATH)
    form  = pd.read_csv(FORM_PATH)

    print(f"  squad_features : {len(squad)} 隊 × {len(squad.columns)} 欄")
    print(f"  team_stats     : {len(stats)} 筆（{stats['period'].nunique()} 個時期）")
    print(f"  form           : {len(form)} 隊")

    # ════════════════════════════════════════════════════════════
    # 1. 從 team_stats 拆出三個時期，各自 pivot 成寬表
    #    欄位命名規則：{指標}_{時期縮寫}
    #    all           → _all
    #    recent_5yr    → _r5     ← 最重要，近期表現
    #    WC_finals_only→ _wc     ← 世界盃大賽成績
    # ════════════════════════════════════════════════════════════
    print("\n🔧 處理 team_stats...")

    stat_cols = ["win_rate", "goals_for_avg", "goals_against_avg", "goal_diff_avg"]
    period_suffix = {
        "all":            "_all",
        "recent_5yr":     "_r5",
        "WC_finals_only": "_wc",
    }

    stats_wide = None
    for period, suffix in period_suffix.items():
        sub = (stats[stats["period"] == period]
               [["team"] + stat_cols]
               .rename(columns={c: c + suffix for c in stat_cols})
               .reset_index(drop=True))

        if stats_wide is None:
            stats_wide = sub
        else:
            stats_wide = stats_wide.merge(sub, on="team", how="outer")

    # 沒有 WC 決賽圈紀錄的 4 隊（Cape Verde / Curaçao / Jordan / Uzbekistan）
    # _wc 欄位會是 NaN，用 _r5 的值填補（近期表現作為替代）
    wc_cols = [c for c in stats_wide.columns if c.endswith("_wc")]
    r5_cols = [c.replace("_wc", "_r5") for c in wc_cols]
    for wc_col, r5_col in zip(wc_cols, r5_cols):
        stats_wide[wc_col] = stats_wide[wc_col].fillna(stats_wide[r5_col])

    print(f"  stats_wide：{len(stats_wide)} 隊 × {len(stats_wide.columns)} 欄")
    print(f"  欄位：{stats_wide.columns.tolist()}")

    # ════════════════════════════════════════════════════════════
    # 2. 從 form 取需要的欄位
    #    form_score_official：只算正式賽事（去掉友誼賽）的近10場加權分
    #    比 form_score_all 更能反映大賽備戰狀態
    # ════════════════════════════════════════════════════════════
    print("\n🔧 處理 form...")

    form_clean = form[["team", "form_score_all", "form_score_official"]].copy()
    print(f"  form_score_official 範圍：{form_clean['form_score_official'].min():.3f} ~ {form_clean['form_score_official'].max():.3f}")

    # ════════════════════════════════════════════════════════════
    # 3. 合併所有特徵
    # ════════════════════════════════════════════════════════════
    print("\n🔗 合併特徵表...")

    final = (squad
             .merge(stats_wide, on="team", how="left")
             .merge(form_clean,  on="team", how="left"))

    # ════════════════════════════════════════════════════════════
    # 4. 欄位排序（方便後續閱讀與建模）
    # ════════════════════════════════════════════════════════════
    col_order = [
        "team",
        # ── 身價 / 陣容結構 ──
        "total_value_eur", "avg_age", "total_caps", "total_goals",
        "top_club_ratio", "attack_value_ratio",
        # ── EA 能力值 ──
        "game_top_11_ovr", "game_fw_speed", "game_fw_finishing",
        "game_mf_passing", "game_df_defense", "game_df_physic",
        # ── 歷史勝率（全期） ──
        "win_rate_all", "goals_for_avg_all", "goals_against_avg_all", "goal_diff_avg_all",
        # ── 近5年表現（最重要） ──
        "win_rate_r5", "goals_for_avg_r5", "goals_against_avg_r5", "goal_diff_avg_r5",
        # ── 世界盃決賽圈成績 ──
        "win_rate_wc", "goals_for_avg_wc", "goals_against_avg_wc", "goal_diff_avg_wc",
        # ── 近期狀態 ──
        "form_score_all", "form_score_official",
    ]
    # 只取存在的欄位（避免欄位缺漏報錯）
    final = final[[c for c in col_order if c in final.columns]]

    # ════════════════════════════════════════════════════════════
    # 5. 缺值檢查
    # ════════════════════════════════════════════════════════════
    null_counts = final.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"\n⚠️  有缺值的欄位：")
        print(null_cols.to_string())
    else:
        print("\n✅ 無缺值")

    # ════════════════════════════════════════════════════════════
    # 6. 儲存
    # ════════════════════════════════════════════════════════════
    final.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n💾 儲存：{OUT_PATH}")
    print(f"   {len(final)} 隊 × {len(final.columns)} 欄")

    # 預覽：各維度 top3
    print("\n📊 預覽各維度 top3：")
    preview_cols = {
        "game_top_11_ovr":    "EA 整體戰力",
        "total_value_eur":    "總身價",
        "win_rate_r5":        "近5年勝率",
        "form_score_official":"近期狀態",
        "win_rate_wc":        "WC決賽圈勝率",
    }
    for col, label in preview_cols.items():
        if col in final.columns:
            top3 = final.nlargest(3, col)[["team", col]]
            teams_str = ", ".join(f"{r['team']}({r[col]:.3f})" for _, r in top3.iterrows())
            print(f"  {label:15s}：{teams_str}")


if __name__ == "__main__":
    build_final_features()
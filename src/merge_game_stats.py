import pandas as pd
import numpy as np

# ── 路徑設定 ────────────────────────────────────────────────────
# 直接讀 clean_ea_players.py 已產出的檔案，不重新比對
SQUADS_EA_PATH  = r"data/processed/wc26_squads_ea.csv"
SQUAD_FEAT_PATH = r"data/processed/wc26_squad_features.csv"
OUT_FEAT_PATH   = r"data/processed/wc26_squad_features.csv"
# ────────────────────────────────────────────────────────────────

# 欄位對應（clean_ea_players.py 輸出的小寫欄位名）
# OVR          → ovr
# Sprint Speed → sprint_speed
# Finishing    → finishing
# PAS          → pas
# Def Awareness→ def_awareness
# PHY          → phy


def aggregate_team_features():
    print("📥 讀取 wc26_squads_ea.csv...")
    try:
        sq = pd.read_csv(SQUADS_EA_PATH)
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案：{e}")
        return

    print(f"  球員數：{len(sq)}")

    # ── 去重：同隊同名保留 OVR 最高那筆 ──
    sq_clean = (sq.sort_values("ovr", ascending=False)
                  .drop_duplicates(subset=["Country", "Player"], keep="first")
                  .copy())
    print(f"  去重後：{len(sq_clean)}（移除 {len(sq)-len(sq_clean)} 筆重複）")

    def safe_mean(series):
        """只計算有 EA 資料的球員，NaN 不參與平均"""
        vals = series.dropna()
        return round(vals.mean(), 4) if len(vals) > 0 else np.nan

    print("\n📊 聚合國家隊特徵...")
    team_features = []
    for team, group in sq_clean.groupby("Country"):

        # 位置篩選在 group 內做（避免全局 index 對不上的問題）
        fw_group = group[group["Pos."] == "FW"]
        mf_group = group[group["Pos."] == "MF"]
        df_group = group[group["Pos."] == "DF"]

        # top11 OVR：只取有 EA 資料的球員
        top11_ovr = group["ovr"].dropna().nlargest(11).mean()

        team_features.append({
            "team":              team,
            "game_top_11_ovr":   round(top11_ovr, 4) if pd.notna(top11_ovr) else np.nan,
            "game_fw_speed":     safe_mean(fw_group["sprint_speed"]),    # FW only
            "game_fw_finishing": safe_mean(fw_group["finishing"]),        # FW only
            "game_mf_passing":   safe_mean(mf_group["pas"]),             # MF only
            "game_df_defense":   safe_mean(df_group["def_awareness"]),   # DF only
            "game_df_physic":    safe_mean(df_group["phy"]),             # DF only
        })

    df_team_game = pd.DataFrame(team_features)

    # ── 與既有特徵表合併 ──
    try:
        existing = pd.read_csv(SQUAD_FEAT_PATH)
        # 移除舊的 game_ 欄位，換成重新計算的版本
        existing = existing.loc[:, ~existing.columns.str.startswith("game_")]
        final = pd.merge(existing, df_team_game, on="team", how="left")
        final.to_csv(OUT_FEAT_PATH, index=False, encoding="utf-8-sig")

        print(f"🎉 特徵表已更新：{OUT_FEAT_PATH}")
        print(f"   欄位：{list(final.columns)}")
        print("\n🔥 top5 預覽：")
        print(final.sort_values("game_top_11_ovr", ascending=False)
              [["team", "game_top_11_ovr", "game_fw_speed",
                "game_df_defense", "game_df_physic"]]
              .head(5).to_string(index=False))

    except FileNotFoundError:
        # 如果還沒有既有特徵表，直接輸出 game 特徵
        df_team_game.to_csv(OUT_FEAT_PATH, index=False, encoding="utf-8-sig")
        print(f"🎉 新建特徵表：{OUT_FEAT_PATH}")


if __name__ == "__main__":
    aggregate_team_features()



"""
EA SPORTS FC 25 球員資料清整腳本
==================================
輸入：
  male_players.csv          原始 EA FC 25 資料集
  wc2026_squads_final.csv   大名單（含 player_id / market_value）

輸出：
  ea_players_cleaned.csv    完整清整後的 EA 球員資料（16,161 人）
  wc26_squads_ea.csv        大名單 + EA OVR 及各項數值（合併結果）

執行：
  pip install pandas rapidfuzz
  python clean_ea_players.py
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from rapidfuzz import process, fuzz

# ── 路徑設定 ────────────────────────────────────────────────────
EA_CSV     = "C:/Users/林翰緯/world_cup_prediction/data/raw/male_players.csv"
SQUADS_CSV = "C:/Users/林翰緯/world_cup_prediction/data/processed/wc2026_squads_final.csv"
OUT_EA     = "C:/Users/林翰緯/world_cup_prediction/data/processed/ea_players_cleaned.csv"
OUT_SQUADS = "C:/Users/林翰緯/world_cup_prediction/data/processed/wc26_squads_ea.csv"
# ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# 1. 讀取資料
# ════════════════════════════════════════════════════════════════
print("📥 讀取資料...")
ea = pd.read_csv(EA_CSV, low_memory=False)
squads = pd.read_csv(SQUADS_CSV)
print(f"  EA FC 25 球員：{len(ea):,} 人")
print(f"  世界盃大名單：{len(squads):,} 人")


# ════════════════════════════════════════════════════════════════
# 2. 移除垃圾欄位
# ════════════════════════════════════════════════════════════════
print("\n🗑️  移除垃圾欄位...")

drop_cols = ["Unnamed: 0.1", "Unnamed: 0"]
ea = ea.drop(columns=[c for c in drop_cols if c in ea.columns])


# ════════════════════════════════════════════════════════════════
# 3. 從 URL 抽取 ea_id（唯一識別碼，不受名字編碼問題影響）
#    URL 格式：.../player-ratings/{slug}/{ea_id}
# ════════════════════════════════════════════════════════════════
print("🔑 抽取 ea_id 與 name_slug...")

ea["ea_id"] = ea["url"].str.extract(r"/(\d+)$").astype("Int64")
ea["name_slug"] = ea["url"].str.extract(r"player-ratings/([^/]+)/\d+$")
# name_slug 是 URL 友善格式（無重音、全小寫連字號），可作為乾淨的英文名比對基準


# ════════════════════════════════════════════════════════════════
# 4. 修正 Name 欄位的編碼問題
#    原始 Name 有亂碼（é→矇、ç→癟 等），改用 name_slug 重建可讀名稱
#    策略：slug 轉 Title Case（kylian-mbappe → Kylian Mbappe）
#    注意：這會失去重音，但至少可讀，後續合併以 ea_id 為主不靠名字
# ════════════════════════════════════════════════════════════════
print("✏️  修正 Name 編碼...")

ea["Name_clean"] = ea["name_slug"].str.replace("-", " ").str.title()


# ════════════════════════════════════════════════════════════════
# 5. 身高體重拆分
#    原始：'182cm / 6\'0"'  → height_cm=182
#    原始：'75kg / 165lb'   → weight_kg=75
# ════════════════════════════════════════════════════════════════
print("📐 拆分身高體重...")

ea["height_cm"] = ea["Height"].str.extract(r"(\d+)cm").astype("Int64")
ea["weight_kg"] = ea["Weight"].str.extract(r"(\d+)kg").astype("Int64")
ea = ea.drop(columns=["Height", "Weight"])


# ════════════════════════════════════════════════════════════════
# 6. Nation 隊名標準化（對齊大名單 Country 欄位）
#    EA 用了不同的國名，建立映射表
# ════════════════════════════════════════════════════════════════
print("🌍 標準化 Nation 名稱...")

nation_map = {
    "Holland":           "Netherlands",     # EA 用 Holland
    "Korea Republic":    "South Korea",     # EA 用 Korea Republic
    "Congo DR":          "DR Congo",
    "Cape Verde Islands":"Cape Verde",
    "C\u00f4te d'Ivoire": "Ivory Coast",    # EA 用法文名（含正確 unicode）
    "Cura\u00e7ao":      "Curaçao",
    # 編碼損毀版本也補上（以防讀取時已損毀）
    "C繫te d'Ivoire":    "Ivory Coast",
    "Cura癟ao":          "Curaçao",
}

ea["Nation_std"] = ea["Nation"].replace(nation_map)


# ════════════════════════════════════════════════════════════════
# 7. 欄位重新命名，提高可讀性
# ════════════════════════════════════════════════════════════════
print("📝 重新命名欄位...")

rename_map = {
    "Name":              "name_raw",        # 原始（有編碼問題）保留備查
    "Name_clean":        "name",            # 清整後的英文名
    "OVR":               "ovr",
    "PAC":               "pac",
    "SHO":               "sho",
    "PAS":               "pas",
    "DRI":               "dri",
    "DEF":               "def",
    "PHY":               "phy",
    "Position":          "position",
    "Weak foot":         "weak_foot",
    "Skill moves":       "skill_moves",
    "Preferred foot":    "preferred_foot",
    "Alternative positions": "alt_positions",
    "Age":               "age",
    "Nation":            "nation_raw",
    "Nation_std":        "nation",
    "League":            "league",
    "Team":              "team",
    "play style":        "play_style",
    "Rank":              "ea_rank",
    "Acceleration":      "acceleration",
    "Sprint Speed":      "sprint_speed",
    "Positioning":       "positioning",
    "Finishing":         "finishing",
    "Shot Power":        "shot_power",
    "Long Shots":        "long_shots",
    "Volleys":           "volleys",
    "Penalties":         "penalties",
    "Vision":            "vision",
    "Crossing":          "crossing",
    "Free Kick Accuracy":"free_kick_acc",
    "Short Passing":     "short_passing",
    "Long Passing":      "long_passing",
    "Curve":             "curve",
    "Dribbling":         "dribbling",
    "Agility":           "agility",
    "Balance":           "balance",
    "Reactions":         "reactions",
    "Ball Control":      "ball_control",
    "Composure":         "composure",
    "Interceptions":     "interceptions",
    "Heading Accuracy":  "heading_acc",
    "Def Awareness":     "def_awareness",
    "Standing Tackle":   "standing_tackle",
    "Sliding Tackle":    "sliding_tackle",
    "Jumping":           "jumping",
    "Stamina":           "stamina",
    "Strength":          "strength",
    "Aggression":        "aggression",
    "GK Diving":         "gk_diving",
    "GK Handling":       "gk_handling",
    "GK Kicking":        "gk_kicking",
    "GK Positioning":    "gk_positioning",
    "GK Reflexes":       "gk_reflexes",
}

ea = ea.rename(columns={k: v for k, v in rename_map.items() if k in ea.columns})

# 整理欄位順序
priority_cols = [
    "ea_id", "ea_rank", "name", "name_raw", "name_slug",
    "ovr", "pac", "sho", "pas", "dri", "def", "phy",
    "position", "alt_positions",
    "age", "nation", "nation_raw", "league", "team",
    "preferred_foot", "weak_foot", "skill_moves",
    "height_cm", "weight_kg", "play_style",
]
stat_cols = [
    "acceleration", "sprint_speed", "positioning", "finishing",
    "shot_power", "long_shots", "volleys", "penalties",
    "vision", "crossing", "free_kick_acc", "short_passing",
    "long_passing", "curve", "dribbling", "agility", "balance",
    "reactions", "ball_control", "composure", "interceptions",
    "heading_acc", "def_awareness", "standing_tackle", "sliding_tackle",
    "jumping", "stamina", "strength", "aggression",
    "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
    "url",
]
all_ordered = priority_cols + stat_cols
ea = ea[[c for c in all_ordered if c in ea.columns]]


# ════════════════════════════════════════════════════════════════
# 8. GK 欄位處理
#    只有 GK 位置才有 GK 數值，其他位置填 0 比留 NaN 好統計
# ════════════════════════════════════════════════════════════════
print("🧤 處理 GK 欄位...")

gk_cols = ["gk_diving","gk_handling","gk_kicking","gk_positioning","gk_reflexes"]
ea[gk_cols] = ea[gk_cols].fillna(0).astype(int)


# ════════════════════════════════════════════════════════════════
# 9. 儲存完整清整結果
# ════════════════════════════════════════════════════════════════
print(f"\n💾 儲存 {OUT_EA}...")
ea.to_csv(OUT_EA, index=False, encoding="utf-8-sig")
print(f"  {len(ea):,} 筆，{len(ea.columns)} 欄")


# ════════════════════════════════════════════════════════════════
# 10. 合併至世界盃大名單
#     比對策略（三層）：
#     ① name_slug vs 大名單 player_slug（從大名單 Player 欄產生）
#     ② fuzzy 補救（threshold=88）
#     ③ Nation 交叉驗證（同名不同國時取正確那個）
# ════════════════════════════════════════════════════════════════
print("\n🔗 合併大名單...")

def to_slug(s):
    """轉成 URL slug 格式：去重音、小寫、空格換連字號"""
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s

def normalize(s):
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()

squads["player_slug"] = squads["Player"].apply(to_slug)
squads["player_norm"] = squads["Player"].apply(normalize)

ea["name_norm"] = ea["name"].apply(normalize)

# EA 的 nation → squads 的 Country 對照（反向查）
ea_slug_to_row = ea.set_index("name_slug")

# ── 手動修正表：Wikipedia 名 → EA name_slug ──────────────────────────────
# 三類問題：
#   類型1：EA 用暱稱/縮寫/全名不同（Vinícius Júnior → vini-jr）
#   類型2：Mc/Mac/O' 等前綴在 EA slug 裡被斷開（McTominay → mc-tominay）
#   類型3：北歐/土耳其特殊字元在 slug 裡變成 `-`（ø → -, ı → -)
MANUAL_SLUG_FIXES = {
    # ── 類型1：暱稱 / 縮寫 / 全名不同 ──
    "Vinícius Júnior":       "vini-jr",
    "Neymar":                "neymar-jr",
    "Martín Zubimendi":      "zubimendi",
    "Mikel Oyarzabal":       "oyarzabal",
    "Álex Grimaldo":         "grimaldo",
    "Franck Kessié":         "franck-yannick-kessie",
    "Abde Ezzalzouli":       "abdessamad-ezzalzouli",
    "Julio Enciso":          "julio-cesar-enciso",
    "José Giménez":          "jose-maria-gimenez",
    "Gabriel Magalhães":     "gabriel",
    "Hannibal Mejbri":       "hannibal",
    "Andy Robertson":        "andrew-robertson",
    "Musa Al-Taamari":       "musa-al-tamari",
    "Anis Ben Slimane":      "anis-slimane",
    "Musab Al-Juwayr":       "musab-al-juwair",
    "Alaa Al-Hejji":         "alaa-al-hajji",
    "Moteb Al-Harbi":        "muteb-al-harbi",
    "Ange-Yoan Bonny":       "yoan-bonny",
    "Weverton":              "weverson",
    "Dennis Eckert":         "dennis-eckert-ayensa",
    "Dailon Livramento":     "dailon-rocha-livramento",
    "Hassan Al-Tambakti":    "hassan-tombakti",
    "Firas Al-Buraikan":     "firas-al-birekan",
    # ── 類型2：Mc/Mac/O' 斷字 ──
    "Scott McTominay":       "scott-mc-tominay",
    "Weston McKennie":       "weston-mc-kennie",
    "John McGinn":           "john-mc-ginn",
    "Scott McKenna":         "scott-mc-kenna",
    "Mark McKenzie":         "mark-mc-kenzie",
    "Kenny McLean":          "kenny-mc-lean",
    "Aiden O'Neill":         "aiden-o-neill",
    "Callum McCowatt":       "callum-mc-cowatt",
    # ── 類型3：特殊字元 ──
    "Kenan Yıldız":          "kenan-y-ld-z",
    "Jørgen Strand Larsen":  "j-rgen-strand-larsen",
    "Barış Alper Yılmaz":    "bar-s-alper-y-lmaz",
    "N'Golo Kanté":          "n-golo-kante",
    "Ngal'ayel Mukau":       "ngal-ayel-mukau",
    "Fredrik André Bjørkan": "fredrik-andre-bj-rkan",
    "Torbjørn Heggem":       "torbj-rn-l-heggem",
    "Lee Dong-gyeong":       "lee-dong-kyeong",
    "Kim Jin-gyu":           "kim-jin-kyu",
}

# ── Step 1: slug 完整比對 ──
s1 = squads.merge(
    ea[["ea_id","name_slug","ovr","pac","sho","pas","dri","def","phy",
        "position","height_cm","weight_kg","preferred_foot","weak_foot",
        "skill_moves","nation","play_style"] +
       ["acceleration","sprint_speed","positioning","finishing","shot_power",
        "long_shots","volleys","penalties","vision","crossing","free_kick_acc",
        "short_passing","long_passing","curve","dribbling","agility","balance",
        "reactions","ball_control","composure","interceptions","heading_acc",
        "def_awareness","standing_tackle","sliding_tackle","jumping","stamina",
        "strength","aggression","gk_diving","gk_handling","gk_kicking",
        "gk_positioning","gk_reflexes"]],
    left_on="player_slug", right_on="name_slug", how="left"
)
s1["ea_match"] = s1["ea_id"].apply(lambda x: "slug" if pd.notna(x) else "")

# ── Step 2: name_norm 比對（補救 slug 差異）──
ea_norm_map = ea.set_index("name_norm")[["ea_id","ovr","pac","sho","pas","dri","def","phy",
    "position","height_cm","weight_kg","preferred_foot","weak_foot","skill_moves",
    "nation","play_style","acceleration","sprint_speed","positioning","finishing",
    "shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_acc",
    "short_passing","long_passing","curve","dribbling","agility","balance","reactions",
    "ball_control","composure","interceptions","heading_acc","def_awareness",
    "standing_tackle","sliding_tackle","jumping","stamina","strength","aggression",
    "gk_diving","gk_handling","gk_kicking","gk_positioning","gk_reflexes"]]

# 去重（同 norm 取 OVR 最高）
ea_norm_map = ea_norm_map.sort_values("ovr", ascending=False)
ea_norm_map = ea_norm_map[~ea_norm_map.index.duplicated(keep="first")]

unmatched_mask = s1["ea_match"] == ""
for idx in s1[unmatched_mask].index:
    norm = s1.at[idx, "player_norm"]
    if norm in ea_norm_map.index:
        for col in ea_norm_map.columns:
            s1.at[idx, col] = ea_norm_map.at[norm, col]
        s1.at[idx, "ea_match"] = "norm"

# ── Step 2.5: 手動 slug 修正（暱稱/斷字/特殊字元三類問題）──
ea_slug_to_data = ea.set_index("name_slug")[[
    "ea_id","ovr","pac","sho","pas","dri","def","phy",
    "position","height_cm","weight_kg","preferred_foot","weak_foot",
    "skill_moves","nation","play_style",
    "acceleration","sprint_speed","positioning","finishing","shot_power",
    "long_shots","volleys","penalties","vision","crossing","free_kick_acc",
    "short_passing","long_passing","curve","dribbling","agility","balance",
    "reactions","ball_control","composure","interceptions","heading_acc",
    "def_awareness","standing_tackle","sliding_tackle","jumping","stamina",
    "strength","aggression","gk_diving","gk_handling","gk_kicking",
    "gk_positioning","gk_reflexes"
]]
# 同 slug 取 OVR 最高
ea_slug_to_data = ea_slug_to_data.sort_values("ovr", ascending=False)
ea_slug_to_data = ea_slug_to_data[~ea_slug_to_data.index.duplicated(keep="first")]

manual_fixed = 0
for idx in s1[s1["ea_match"] == ""].index:
    player_name = s1.at[idx, "Player"]
    if player_name in MANUAL_SLUG_FIXES:
        target_slug = MANUAL_SLUG_FIXES[player_name]
        if target_slug in ea_slug_to_data.index:
            for col in ea_slug_to_data.columns:
                if col in s1.columns:
                    s1.at[idx, col] = ea_slug_to_data.at[target_slug, col]
            s1.at[idx, "ea_match"] = "manual"
            manual_fixed += 1

print(f"  手動修正補救：{manual_fixed} 人")

# ── Step 3: fuzzy 補救 ──
ea_norm_list = list(ea_norm_map.index)
still_unmatched = s1[s1["ea_match"] == ""].index

for idx in still_unmatched:
    norm = s1.at[idx, "player_norm"]
    country = s1.at[idx, "Country"]
    match = process.extractOne(norm, ea_norm_list,
                               scorer=fuzz.token_sort_ratio, score_cutoff=88)
    if match:
        candidate_nation = ea_norm_map.at[match[0], "nation"]
        # Nation 交叉驗證：避免同名不同國的錯誤比對
        if candidate_nation == country or pd.isna(candidate_nation):
            for col in ea_norm_map.columns:
                s1.at[idx, col] = ea_norm_map.at[match[0], col]
            s1.at[idx, "ea_match"] = f"fuzzy_{match[1]:.0f}"

# ── 清理輸出欄位 ──
drop = ["player_slug","player_norm","name_slug"]
result = s1.drop(columns=[c for c in drop if c in s1.columns])

# ── 統計 ──
total = len(result)
matched_ea = (result["ea_match"] != "").sum()
no_ea = (result["ea_match"] == "").sum()

print(f"\n  總球員：{total}")
print(f"  成功取得 EA OVR：{matched_ea} ({matched_ea/total*100:.1f}%)")
print(f"  無 EA 資料：{no_ea} ({no_ea/total*100:.1f}%)")
print(f"\n  比對方法分布：")
print(result["ea_match"].value_counts().to_string())

print("\n  OVR 前10：")
top10 = result[result["ovr"].notna()].nlargest(10, "ovr")[["Player","Country","Pos.","ovr","market_value_M"]]
print(top10.to_string(index=False))

result.to_csv(OUT_SQUADS, index=False, encoding="utf-8-sig")
print(f"\n💾 儲存 {OUT_SQUADS} 完成")

print("\n" + "="*52)
print("✅ 清整完成！")
print(f"  {OUT_EA}     — EA FC 25 完整清整資料（{len(ea):,} 人）")
print(f"  {OUT_SQUADS} — 大名單 + EA OVR 合併結果")
print()
print("📌 合併後可用特徵：")
print("  球員層級：ovr, pac, sho, pas, dri, def, phy + 29項細分數值")
print("  國家隊層級：先發11人 OVR 均值、各位置群組均值")
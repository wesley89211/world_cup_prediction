"""
wc2026_report.py
================
2026 FIFA 世界盃 — 多主題資料分析報告
從 wc26_final_features.csv 產出純文字排行榜

執行：
    python src/wc2026_report.py

輸出：
    終端機彩色報告 + data/reports/wc2026_report.txt（純文字存檔）
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────────────
FEATURES_PATH = r"data/processed/wc26_final_features.csv"
OUT_DIR       = Path("data/reports")
OUT_TXT       = OUT_DIR / "wc2026_report.txt"
# ─────────────────────────────────────────────────────────────────

# ── 終端機色碼（可選）────────────────────────────────────────────
GOLD   = "\033[33m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    # 洲別對應
    continent_map = {
        "Brazil":"南美洲","Argentina":"南美洲","Uruguay":"南美洲","Colombia":"南美洲",
        "Ecuador":"南美洲","Venezuela":"南美洲","Chile":"南美洲","Paraguay":"南美洲",
        "Canada":"北美洲","USA":"北美洲","United States":"北美洲","Mexico":"北美洲",
        "Panama":"北美洲","Costa Rica":"北美洲","Honduras":"北美洲","Jamaica":"北美洲",
        "Haiti":"北美洲","Curaçao":"北美洲","Trinidad and Tobago":"北美洲",
        "France":"歐洲","England":"歐洲","Spain":"歐洲","Germany":"歐洲",
        "Portugal":"歐洲","Netherlands":"歐洲","Belgium":"歐洲","Croatia":"歐洲",
        "Italy":"歐洲","Switzerland":"歐洲","Austria":"歐洲","Scotland":"歐洲",
        "Turkey":"歐洲","Serbia":"歐洲","Ukraine":"歐洲","Norway":"歐洲",
        "Slovakia":"歐洲","Albania":"歐洲","Bosnia and Herzegovina":"歐洲",
        "Czech Republic":"歐洲","Sweden":"歐洲",
        "Uzbekistan":"亞洲","South Korea":"亞洲","Japan":"亞洲","Iran":"亞洲",
        "Saudi Arabia":"亞洲","Australia":"亞洲","Jordan":"亞洲","Qatar":"亞洲","Iraq":"亞洲",
        "Morocco":"非洲","Senegal":"非洲","Egypt":"非洲","Ivory Coast":"非洲",
        "Algeria":"非洲","Ghana":"非洲","Nigeria":"非洲","South Africa":"非洲",
        "DR Congo":"非洲","Cape Verde":"非洲","Cameroon":"非洲","Tanzania":"非洲","Tunisia":"非洲",
        "New Zealand":"大洋洲",
    }
    df["continent"] = df["team"].map(continent_map).fillna("其他")
    df["total_value_M"] = (df["total_value_eur"] / 1e6).round(1)
    return df


# ── 工具函式 ─────────────────────────────────────────────────────

def rank_bar(value, max_val, width=20):
    """產生 ASCII 進度條"""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)

def medal(rank):
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f" {rank:2d}.")

def fmt_val(v, decimals=1, suffix=""):
    if pd.isna(v):
        return "  N/A"
    return f"{v:>{7}.{decimals}f}{suffix}"

def section(title, lines):
    """回傳一個段落的純文字"""
    sep = "─" * 56
    return f"\n{sep}\n  {title}\n{sep}\n" + "\n".join(lines) + "\n"

def top_n(df, col, n=5, ascending=False, fmt=".1f", suffix="", label=None):
    """從 df 取前 n 名，回傳格式化列表"""
    valid = df.dropna(subset=[col]).copy()
    sorted_df = valid.sort_values(col, ascending=ascending).reset_index(drop=True)
    max_val = sorted_df[col].max() if not ascending else sorted_df[col].min()
    max_val = sorted_df[col].abs().max()
    lines = []
    for i, row in sorted_df.head(n).iterrows():
        val = row[col]
        bar = rank_bar(abs(val), max_val)
        val_str = f"{val:{fmt}}{suffix}"
        lines.append(
            f"  {medal(i+1)} {row['team']:<28s} {val_str:>10s}  {bar}"
        )
    return lines

def bottom_n(df, col, n=5, fmt=".1f", suffix="", label=None):
    """取後 n 名（最高值，即排名最差的）"""
    valid = df.dropna(subset=[col]).copy()
    sorted_df = valid.sort_values(col, ascending=False).reset_index(drop=True)
    max_val = sorted_df[col].abs().max()
    lines = []
    for i, row in sorted_df.head(n).iterrows():
        val = row[col]
        bar = rank_bar(abs(val), max_val)
        val_str = f"{val:{fmt}}{suffix}"
        lines.append(
            f"  {medal(i+1)} {row['team']:<28s} {val_str:>10s}  {bar}"
        )
    return lines


# ── 各主題分析函式 ────────────────────────────────────────────────

def section_ea_ovr(df):
    lines = ["  ▸ 依 EA FC 25 先發最強 11 人平均 OVR 排名",
             "  ▸ 反映球隊個人天花板戰力\n"]
    lines += top_n(df, "game_top_11_ovr", 10, fmt=".2f")
    return section("🎮  EA 整體戰力 TOP 10（game_top_11_ovr）", lines)

def section_market_value(df):
    lines = ["  ▸ 全隊球員 Transfermarkt 最新身價加總",
             "  ▸ 資金深度 ≈ 球員質量的市場共識\n"]
    lines += top_n(df, "total_value_M", 10, fmt=".0f", suffix=" M€")
    return section("💰  球隊總身價 TOP 10（百萬歐元）", lines)

def section_win_rate_recent(df):
    lines = ["  ▸ 2021 年以來所有正式 + 友誼賽勝率",
             "  ▸ 最能反映當下球隊狀態\n"]
    lines += top_n(df, "win_rate_r5", 10, fmt=".3f", suffix="")
    lines += ["\n  ▸ 後段班（近5年最低勝率）\n"]
    lines += top_n(df, "win_rate_r5", 5, ascending=True, fmt=".3f")
    return section("📈  近5年勝率排行（2021—2026）", lines)

def section_wc_record(df):
    lines = ["  ▸ 歷屆世界盃決賽圈（1930—2022）勝率",
             "  ▸ 大賽經驗與抗壓能力的歷史指標",
             "  ▸ ⚠ 只計入曾實際出賽的隊伍（排除填補值）\n"]

    # 讀 team_stats 取得真實有 WC_finals_only 紀錄的隊伍清單
    stats_paths = ["data/processed/wc26_team_stats.csv", "wc26_team_stats.csv"]
    wc_teams_real = None
    for p in stats_paths:
        try:
            ts = pd.read_csv(p)
            wc_teams_real = set(ts[ts["period"]=="WC_finals_only"]["team"])
            break
        except FileNotFoundError:
            continue

    if wc_teams_real:
        df_wc = df[df["team"].isin(wc_teams_real)].copy()
    else:
        df_wc = df.copy()

    lines += top_n(df_wc, "win_rate_wc", 10, fmt=".3f")
    lines += ["\n  ▸ 首次或幾乎無決賽圈紀錄的隊伍\n"]
    no_win = df[df["team"].isin(wc_teams_real) & (df["win_rate_wc"] == 0)] if wc_teams_real else df[df["win_rate_wc"]==0]
    never = df[~df["team"].isin(wc_teams_real)] if wc_teams_real else pd.DataFrame()
    for _, row in no_win.iterrows():
        lines.append(f"       {row['team']:<30s} 曾出賽但無勝場")
    for _, row in never.iterrows():
        lines.append(f"       {row['team']:<30s} 首次晉級決賽圈")
    return section("🏆  世界盃決賽圈歷史勝率 TOP 10", lines)

def section_form(df):
    lines = ["  ▸ 近10場正式賽事加權狀態分（越近權重越高）",
             "  ▸ decay=0.85，最能反映當下熱度\n"]
    lines += top_n(df, "form_score_official", 10, fmt=".3f")
    return section("🔥  近期狀態分 TOP 10（正式賽事）", lines)

def section_attack(df):
    lines = ["  ▸ 近5年每場平均進球數",
             "  ▸ 攻擊火力客觀指標（含主客場）\n"]
    lines += top_n(df, "goals_for_avg_r5", 10, fmt=".3f", suffix=" 球/場")
    return section("⚽  近5年進攻火力 TOP 10（平均進球/場）", lines)

def section_defense(df):
    lines = ["  ▸ 近5年每場平均失球數（越低越好）",
             "  ▸ 防守穩定性客觀指標\n"]
    lines += top_n(df, "goals_against_avg_r5", 5, ascending=True, fmt=".3f", suffix=" 球/場")
    lines += ["\n  ▸ 失球最多（最不穩定後防）\n"]
    lines += bottom_n(df, "goals_against_avg_r5", 5, fmt=".3f", suffix=" 球/場")
    return section("🛡️   近5年防守穩定性（平均失球/場）", lines)

def section_goal_diff(df):
    lines = ["  ▸ 近5年（進球 - 失球）每場均值",
             "  ▸ 正負差是衡量綜合競爭力最直接的指標\n"]
    lines += top_n(df, "goal_diff_avg_r5", 10, fmt="+.3f")
    return section("📊  近5年得失球差 TOP 10（每場均值）", lines)

def section_ea_position(df):
    MIN_PLAYERS = 3
    sq_path_candidates = [
        "data/processed/wc26_squads_ea.csv",
        "wc26_squads_ea.csv",
    ]
    sq = None
    for p in sq_path_candidates:
        try:
            sq = pd.read_csv(p)
            break
        except FileNotFoundError:
            continue

    lines = ["  ▸ 各位置群組 EA 屬性排行",
             f"  ▸ 只計算同位置有 \u2265{MIN_PLAYERS} 名球員有 EA 資料的隊伍\n"]

    def pos_rank(pos, attr, n=5):
        if sq is None:
            return ["  （找不到球員層級資料，略過）"]
        grp = (sq[sq["Pos."] == pos]
               .dropna(subset=[attr])
               .groupby("Country")[attr]
               .agg(["mean","count"])
               .reset_index())
        grp = grp[grp["count"] >= MIN_PLAYERS].copy()
        grp = grp.rename(columns={"Country":"team","mean":attr})
        grp = grp.sort_values(attr, ascending=False).reset_index(drop=True)
        max_val = grp[attr].max()
        result = []
        for i, row in grp.head(n).iterrows():
            val = row[attr]
            cnt = int(row["count"])
            bar = rank_bar(val, max_val)
            result.append(
                f"  {medal(i+1)} {row['team']:<28s} {val:>6.1f}  (n={cnt})  {bar}"
            )
        return result

    lines.append("  【鋒線速度 sprint_speed — FW only】")
    lines += pos_rank("FW", "sprint_speed")

    lines.append("\n  【鋒線射門 finishing — FW only】")
    lines += pos_rank("FW", "finishing")

    lines.append("\n  【中場傳球 short_passing — MF only】")
    lines += pos_rank("MF", "short_passing")

    lines.append("\n  【後防意識 def_awareness — DF only】")
    lines += pos_rank("DF", "def_awareness")

    return section("🎯  EA 各位置屬性排行（前5，n=樣本數）", lines)

def section_value_performance(df):
    """超值隊：高勝率低身價"""
    lines = ["  ▸ 近5年勝率 ÷ 身價（每百萬€換來多少勝率）",
             "  ▸ 數字越高 = 低成本高表現 = 黑馬潛力股\n"]
    df2 = df[df["total_value_M"] > 5].copy()
    df2["value_eff"] = df2["win_rate_r5"] / df2["total_value_M"] * 100
    lines += top_n(df2, "value_eff", 8, fmt=".4f", suffix=" 勝率/M€×100")
    return section("💡  超值黑馬指數（勝率 ÷ 身價）", lines)

def section_continent_summary(df):
    lines = ["  ▸ 各洲球隊的平均能力值對比\n"]
    cont_stats = df.groupby("continent").agg(
        隊數=("team","count"),
        平均OVR=("game_top_11_ovr","mean"),
        平均身價M=("total_value_M","mean"),
        近5年勝率=("win_rate_r5","mean"),
        近期狀態=("form_score_official","mean"),
    ).round(2).sort_values("平均OVR", ascending=False)

    header = f"  {'洲別':<10s}  {'隊數':>4s}  {'平均OVR':>8s}  {'平均身價M€':>10s}  {'近5年勝率':>9s}  {'近期狀態':>8s}"
    lines.append(header)
    lines.append("  " + "─" * 58)
    for cont, row in cont_stats.iterrows():
        ovr = f"{row['平均OVR']:.1f}" if not pd.isna(row['平均OVR']) else "  N/A"
        lines.append(
            f"  {cont:<10s}  {int(row['隊數']):>4d}  {ovr:>8s}  "
            f"{row['平均身價M']:>10.0f}  {row['近5年勝率']:>9.3f}  {row['近期狀態']:>8.3f}"
        )
    return section("🌍  洲際競爭力對比", lines)

def section_notable_facts(df):
    """幾個有趣的客觀事實"""
    lines = []

    # 最年輕/最年長陣容
    youngest = df.nsmallest(1, "avg_age").iloc[0]
    oldest   = df.nlargest(1, "avg_age").iloc[0]
    lines.append(f"  最年輕平均年齡：{youngest['team']}（{youngest['avg_age']:.1f} 歲）")
    lines.append(f"  最年長平均年齡：{oldest['team']}（{oldest['avg_age']:.1f} 歲）")

    # 最多經驗值（Caps 總和）
    most_caps = df.nlargest(1, "total_caps").iloc[0]
    lines.append(f"\n  陣容合計 Caps 最多：{most_caps['team']}（{int(most_caps['total_caps'])} caps）")

    # 最多進球數（歷史累積）
    most_goals = df.nlargest(1, "total_goals").iloc[0]
    lines.append(f"  陣容合計歷史進球最多：{most_goals['team']}（{int(most_goals['total_goals'])} 球）")

    # 五大聯賽球員佔比最高
    most_top = df.nlargest(1, "top_club_ratio").iloc[0]
    lines.append(f"\n  五大聯賽球員佔比最高：{most_top['team']}（{most_top['top_club_ratio']*100:.0f}%）")

    # 身價最高鋒線進球力
    best_fw_finish = df.dropna(subset=["game_fw_finishing"]).nlargest(1, "game_fw_finishing").iloc[0]
    lines.append(f"  EA 鋒線射門能力最強：{best_fw_finish['team']}（{best_fw_finish['game_fw_finishing']:.1f}）")

    # 近5年失球最少
    best_def = df.dropna(subset=["goals_against_avg_r5"]).nsmallest(1, "goals_against_avg_r5").iloc[0]
    lines.append(f"\n  近5年失球最少：{best_def['team']}（每場 {best_def['goals_against_avg_r5']:.3f} 球）")

    # 最大驚奇：低身價高勝率
    df2 = df[df["total_value_M"] < 100].dropna(subset=["win_rate_r5"])
    surprise = df2.nlargest(1, "win_rate_r5").iloc[0]
    lines.append(f"  低身價高勝率黑馬：{surprise['team']}（身價 {surprise['total_value_M']:.0f}M€，近5年勝率 {surprise['win_rate_r5']:.3f}）")

    return section("✨  值得注意的客觀事實", lines)


# ── 主程式 ────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}載入資料...{RESET}")
    df = load_data()
    print(f"✅ 共 {len(df)} 支國家隊，{len(df.columns)} 個特徵欄位\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 60
    header = (
        f"\n{SEP}\n"
        "  2026 FIFA 世界盃 — 48 隊多維度資料分析報告\n"
        "  資料來源：Wikipedia / Transfermarkt / EA FC 25 / 歷史賽事\n"
        f"{SEP}"
    )

    sections = [
        header,
        section_ea_ovr(df),
        section_market_value(df),
        section_win_rate_recent(df),
        section_goal_diff(df),
        section_attack(df),
        section_defense(df),
        section_wc_record(df),
        section_form(df),
        section_ea_position(df),
        section_value_performance(df),
        section_continent_summary(df),
        section_notable_facts(df),
        f"\n{'=' * 60}\n  報告結束\n{'=' * 60}\n",
    ]

    full_report = "\n".join(sections)

    # 終端機輸出
    print(full_report)

    # 儲存純文字（去掉色碼）
    import re
    clean = re.sub(r'\033\[[0-9;]*m', '', full_report)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(clean)

    print(f"\n{GREEN}💾 報告已儲存：{OUT_TXT}{RESET}\n")


if __name__ == "__main__":
    main()
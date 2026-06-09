import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="檢查球員名單中的身價缺漏")
    parser.add_argument(
        "-i",
        "--input",
        default="data/processed/wc2026_squads_final.csv",
        help="輸入 CSV 檔案，預設為 data/processed/wc2026_squads_final.csv",
    )
    parser.add_argument(
        "-t",
        "--teams",
        nargs="+",
        help="要檢查的隊伍名稱。可輸入多個，如 Spain Brazil。若省略則檢查所有隊伍。",
    )
    parser.add_argument(
        "--output",
        help="若指定，將漏網之魚輸出到 CSV 檔案。",
    )
    parser.add_argument(
        "--value-col",
        default="market_value_eur",
        help="身價欄位名稱，預設為 market_value_eur。",
    )
    parser.add_argument(
        "--country-col",
        default="Country",
        help="國家欄位名稱，預設為 Country。",
    )
    parser.add_argument(
        "--player-col",
        default="Player",
        help="球員名稱欄位，預設為 Player。",
    )
    parser.add_argument(
        "--club-col",
        default="Club",
        help="球隊欄位名稱，預設為 Club。",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"讀取 CSV 時發生錯誤：{exc}") from exc


def filter_teams(df: pd.DataFrame, country_col: str, teams: list[str] | None) -> pd.DataFrame:
    if teams is None:
        return df

    pattern = "|".join(map(str, teams))
    filtered = df[df[country_col].astype(str).str.contains(pattern, case=False, na=False)]
    return filtered


def find_missing_values(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df[df[value_col].isna() | (df[value_col] == 0)]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    print("🔍 啟動球員身價漏網之魚診斷...")

    try:
        df = load_data(input_path)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    required_cols = [args.country_col, args.player_col, args.club_col, args.value_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print("❌ 下列欄位不存在於輸入資料中：")
        for col in missing_cols:
            print(f"  - {col}")
        print("請確認欄位名稱或使用對應的 --value-col / --country-col / --player-col / --club-col。")
        print(f"可用欄位：{', '.join(df.columns)}")
        return 1

    selected = filter_teams(df, args.country_col, args.teams)
    if args.teams and selected.empty:
        print(f"❌ 找不到符合隊伍：{' '.join(args.teams)}。請確認隊伍名稱是否正確。")
        return 1

    missing = find_missing_values(selected, args.value_col)

    target_label = (
        f"隊伍：{' '.join(args.teams)}" if args.teams else "全部隊伍"
    )
    print(f"\n🔎 已檢查：{target_label}，資料來源：{input_path}")
    print(f"🔢 總共 {len(selected):,} 位球員，缺漏身價：{len(missing):,} 位")

    print("\n📋 找不到身價 (空值或 0) 的球員名單：")
    if missing.empty:
        print("✅ 太神奇了，居然沒有漏網之魚！")
    else:
        print(missing[[args.country_col, args.player_col, args.club_col, args.value_col]].to_string(index=False))
        if args.output:
            output_path = Path(args.output)
            missing.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 已將結果儲存至：{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

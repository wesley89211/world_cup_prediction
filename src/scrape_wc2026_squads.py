"""
2026 FIFA World Cup Squads Scraper
從 Wikipedia 爬取所有 48 支國家隊的球員名單

執行方式：
    pip install requests beautifulsoup4 pandas lxml
    python scrape_wc2026_squads.py

輸出：wc2026_squads.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# ── 關鍵設定 ──────────────────────────────────────────────────────────────────
URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

# Wikipedia 403 的最常見原因：沒有 User-Agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
# ─────────────────────────────────────────────────────────────────────────────


def fetch_page(url: str, retries: int = 3) -> BeautifulSoup:
    """抓頁面，自動重試，回傳 BeautifulSoup 物件"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            print(f"✅ 抓取成功 (HTTP {resp.status_code})，頁面大小 {len(resp.text):,} bytes")
            return BeautifulSoup(resp.text, "lxml")
        except requests.HTTPError as e:
            print(f"⚠️  第 {attempt} 次失敗：{e}")
            if attempt < retries:
                time.sleep(3)
    raise RuntimeError("無法抓取頁面，請確認網路與 headers 設定")


def parse_squads(soup: BeautifulSoup) -> list[dict]:
    """
    解析各國隊伍區塊。

    Wikipedia squad 頁面結構：
      <h2> → 國家名（Group A / Group B ... 或直接是國家）
      <h3> → 國家名（若 h2 是 Group）
      <table class="wikitable"> → 球員名單表格
    """
    records = []
    current_country = None

    # 找 content div 內的所有元素
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        content = soup  # fallback

    elements = content.find_all(["h2", "h3", "table"])

    for el in elements:
        # ── 偵測國家標題 ──
        if el.name in ("h2", "h3"):
            # headline 文字（去掉 [edit] 之類雜訊）
            headline = el.find(class_="mw-headline") or el
            country_text = headline.get_text(" ", strip=True)
            # 跳過非國家標題（Contents / Notes / References 等）
            skip_keywords = {
                "contents", "notes", "references", "external links",
                "statistics", "group a", "group b", "group c", "group d",
                "group e", "group f", "group g", "group h", "group i",
                "group j", "group k", "group l",
            }
            if country_text.lower() in skip_keywords:
                # Group X 標題本身不是國家，清空 current_country
                if country_text.lower().startswith("group"):
                    current_country = None
                continue
            # h2 = Group 或國家，h3 = 國家（在 Group 之下）
            current_country = country_text
            continue

        # ── 解析球員表格 ──
        if el.name == "table" and current_country:
            rows = el.find_all("tr")
            if len(rows) < 2:
                continue

            # 取表頭
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            # 若找不到 "Player" 欄位，跳過（可能是別的表格）
            if not any("Player" in h or "player" in h for h in headers):
                continue

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    continue

                row_data = [c.get_text(" ", strip=True) for c in cells]

                # 對齊欄位數量
                record = {"Country": current_country}
                for i, col in enumerate(headers):
                    record[col] = row_data[i] if i < len(row_data) else ""

                # 清理球員名稱（去掉 "(captain)" 括號）
                for key in list(record.keys()):
                    if "player" in key.lower():
                        record[key] = re.sub(r"\(.*?\)", "", record[key]).strip()

                records.append(record)

    return records


def main():
    print(f"📥 正在抓取：{URL}\n")
    soup = fetch_page(URL)

    print("\n🔍 解析球員資料中...")
    records = parse_squads(soup)

    if not records:
        print("❌ 沒有解析到任何資料，請檢查頁面結構是否改變")
        return

    df = pd.DataFrame(records)

    # 基本清理
    df = df.dropna(how="all")
    df = df[df["Country"].notna()]

    # 統計
    country_count = df["Country"].nunique()
    print(f"\n✅ 共解析到 {len(df)} 筆球員資料，涵蓋 {country_count} 支國家隊")

    # 儲存
    out_path = "wc2026_squads.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig 在 Excel 開啟不會亂碼
    print(f"💾 已儲存至 {out_path}")

    # 預覽
    print("\n📋 前 10 筆資料：")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

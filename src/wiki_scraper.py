import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_all_wiki_squads_statemachine(url):
    """使用狀態機模式：由上而下掃描，完美避開隱形 div 陷阱"""
    logging.info(f"開始向 {url} 發送請求...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        all_players_list = []
        current_team = None  # 記住目前掃描到的國家
        
        # 鎖定維基百科文章的主體區塊
        content_div = soup.find('div', class_='mw-parser-output')
        if not content_div:
            logging.error("找不到文章主體")
            return None

        # 從頭到尾一行一行掃描
        for element in content_div.children:
            if element.name is None:
                continue
                
            # 如果看到標題 (h2 或 h3)，就把裡面的國家名字背起來
            if element.name in ['h2', 'h3']:
                headline = element.find('span', class_='mw-headline')
                if headline:
                    # 濾掉 Group A 這種標題，只抓國家名
                    text = headline.text.replace('[edit]', '').strip()
                    if not text.startswith('Group'):
                        current_team = text
            
            # 處理直接裸露的表格
            elif element.name == 'table' and 'wikitable' in element.get('class', []):
                df = _parse_table(element, current_team)
                if df is not None:
                    all_players_list.append(df)
            
            # 處理被 div 包裝起來的表格 (破解我們剛才踩到的陷阱！)
            elif element.name == 'div':
                tables = element.find_all('table', class_='wikitable')
                for table in tables:
                    df = _parse_table(table, current_team)
                    if df is not None:
                        all_players_list.append(df)
                        
        if all_players_list:
            final_df = pd.concat(all_players_list, ignore_index=True)
            # 清理 [1], [2] 註腳
            final_df.columns = final_df.columns.astype(str).str.replace(r'\[.*\]', '', regex=True)
            logging.info(f"🎉 爬蟲大成功！共抓取了 {len(final_df)} 名球員。")
            return final_df
        else:
            logging.warning("沒有抓到任何名單，請檢查。")
            return None

    except Exception as e:
        logging.error(f"連線或解析失敗: {e}")
        return None

def _parse_table(table_element, team_name):
    """將 HTML table 轉為 DataFrame 的輔助函式"""
    try:
        df = pd.read_html(str(table_element))[0]
        cols_str = " ".join([str(c) for c in df.columns])
        # 確認這真的是球員名單表
        if 'Pos' in cols_str or 'Player' in cols_str:
            if team_name:
                df['National_Team'] = team_name
                logging.info(f"成功抓取: {team_name}")
                return df
    except Exception:
        pass
    return None

if __name__ == "__main__":
    # 這次我們理直氣壯地使用 2026 年網址！
    WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
    
    squads_df = scrape_all_wiki_squads_statemachine(WIKI_URL)
    
    if squads_df is not None:
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "wiki_squads_2026.csv"
        squads_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"資料已存檔至：{output_path}")
        print("\n=== 資料預覽 ===")
        print(squads_df.head())
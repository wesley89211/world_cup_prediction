import pandas as pd  
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CSV_PATH = Path("data/raw/results.csv")
DB_PATH = Path("data/world_cup.db")

def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """執行 Pandas 資料清理 (Data Cleaning)"""
    initial_len = len(df)
    
    # 1. 轉換日期格式
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 2. 移除缺失值 (針對關鍵預測欄位)
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
    
    # 3. 移除重複資料
    df = df.drop_duplicates()
    
    # 4. 資料型態轉換
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    # 5. 球隊名稱一致性處理 (去除首尾空白、轉首字母大寫)
    df['home_team'] = df['home_team'].str.strip().str.title()
    df['away_team'] = df['away_team'].str.strip().str.title()
    
    logging.info(f"資料清理完成：移除了 {initial_len - len(df)} 筆異常/重複資料。剩餘 {len(df)} 筆。")
    return df

def load_data_to_sqlite():
    """執行 ETL 載入程序"""
    if not CSV_PATH.exists():
        logging.error(f"找不到檔案：{CSV_PATH}。請確認是否已下載並放置於正確路徑。")
        return

    # Extract
    logging.info("開始讀取原始 CSV 資料...")
    df = pd.read_csv(CSV_PATH)
    
    # Transform
    df_clean = clean_and_transform_data(df)
    
    # 取得所有不重複的球隊名單 (聯集主隊與客隊)
    unique_teams = pd.concat([df_clean['home_team'], df_clean['away_team']]).unique()
    teams_df = pd.DataFrame({'team_name': unique_teams})
    
    # Load
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # --- 步驟 A：更新維度表 (teams) ---
            logging.info("正在更新球隊維度表 (teams)...")
            for team in unique_teams:
                cursor.execute(
                    "INSERT OR IGNORE INTO teams (team_name) VALUES (?)", 
                    (team,)
                )
            conn.commit()
            
            # --- 步驟 B：取得 team_name 與 team_id 的對應字典 ---
            cursor.execute("SELECT team_id, team_name FROM teams")
            team_mapping = {row[1]: row[0] for row in cursor.fetchall()}
            
            # 將 DataFrame 的文字名稱轉換為資料庫的 team_id
            df_clean['home_team_id'] = df_clean['home_team'].map(team_mapping)
            df_clean['away_team_id'] = df_clean['away_team'].map(team_mapping)
            
            # --- 步驟 C：準備插入事實表 (matches) ---
            # 整理成與資料庫 Table Schema 一致的欄位
            matches_to_insert = df_clean[['date', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'tournament']]
            matches_to_insert = matches_to_insert.rename(columns={'date': 'match_date', 'tournament': 'tournament_type'})
            
            logging.info("正在寫入賽事事實表 (matches)...")
            matches_to_insert.to_sql('matches', conn, if_exists='append', index=False)
            
            logging.info(" ETL 流程全部執行完畢！")
            
    except Exception as e:
        logging.error(f"資料庫寫入失敗：{e}")

if __name__ == "__main__":
    load_data_to_sqlite()
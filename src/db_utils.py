import sqlite3
import logging
from pathlib import Path

# 設定基本日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = Path("data/world_cup.db")

def get_db_connection():
    """建立並回傳 SQLite 資料庫連線"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn

def init_database():
    """初始化資料庫與建立資料表"""
    create_teams_table = """
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_name TEXT NOT NULL UNIQUE,
        confederation TEXT,
        fifa_rank INTEGER
    );
    """
    
    create_matches_table = """
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date DATE NOT NULL,
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        home_score INTEGER,
        away_score INTEGER,
        tournament_type TEXT,
        FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
        FOREIGN KEY (away_team_id) REFERENCES teams (team_id)
    );
    """
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_teams_table)
            cursor.execute(create_matches_table)
            conn.commit()
            logging.info("資料庫與資料表初始化成功！")
            
            # 插入一筆測試資料
            cursor.execute(
                "INSERT OR IGNORE INTO teams (team_name, confederation, fifa_rank) VALUES (?, ?, ?)",
                ("Brazil", "CONMEBOL", 5)
            )
            conn.commit()
            logging.info("測試資料寫入完成。")
            
    except sqlite3.Error as e:
        logging.error(f"資料庫初始化失敗: {e}")

if __name__ == "__main__":
    init_database()
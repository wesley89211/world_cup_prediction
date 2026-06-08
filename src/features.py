import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DB_PATH = Path("data/world_cup.db")

def get_matches_data():
    query = """
    SELECT m.match_id, m.match_date, t1.team_name as home_team, t2.team_name as away_team,
           m.home_score, m.away_score, m.tournament_type
    FROM matches m
    JOIN teams t1 ON m.home_team_id = t1.team_id
    JOIN teams t2 ON m.away_team_id = t2.team_id
    ORDER BY m.match_date ASC
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)
    return df

def calculate_dynamic_elo(df: pd.DataFrame) -> pd.DataFrame:
    """特徵 4.0：加入主場威壓的 ELO 演算法"""
    logging.info("計算動態 ELO 積分 (包含主場加成)...")
    elo_dict = {}
    home_elos, away_elos = [], []
    HOME_ADVANTAGE = 100  # 主場固定加成 100 分
    
    for index, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        if home not in elo_dict: elo_dict[home] = 1500
        if away not in elo_dict: elo_dict[away] = 1500
        
        home_elo, away_elo = elo_dict[home], elo_dict[away]
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        if row['home_score'] > row['away_score']: s_home, s_away = 1, 0
        elif row['home_score'] == row['away_score']: s_home, s_away = 0.5, 0.5
        else: s_home, s_away = 0, 1
            
        # 計算預期勝率時，若不是中立場地，主隊 ELO 虛擬加上 100 分
        home_effective_elo = home_elo + (HOME_ADVANTAGE if row['is_neutral'] == 0 else 0)
        
        e_home = 1 / (1 + 10 ** ((away_elo - home_effective_elo) / 400))
        e_away = 1 / (1 + 10 ** ((home_effective_elo - away_elo) / 400))
        
        k = 20 * row['tournament_weight']
        
        elo_dict[home] = home_elo + k * (s_home - e_home)
        elo_dict[away] = away_elo + k * (s_away - e_away)
        
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("開始執行特徵工程 4.0...")
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # 1. 基礎目標與賽事權重
    df['result'] = df.apply(lambda r: 2 if r['home_score'] > r['away_score'] else (1 if r['home_score'] == r['away_score'] else 0), axis=1)
    df['tournament_weight'] = df['tournament_type'].apply(lambda x: 3 if any(t in x for t in ['World Cup', 'Copa America', 'Euro']) else (1 if 'Friendly' in x else 2))
    df['is_neutral'] = df['tournament_type'].apply(lambda x: 1 if 'World Cup' in x else 0)
    
    # 2. 算入動態 ELO 積分 (已包含主場加成)
    df = calculate_dynamic_elo(df)
    
    # 3. 處理時間序列特徵 (疲勞度與零封率)
    logging.info("計算球隊疲勞度與防守穩固度...")
    
    home_df = df[['match_date', 'home_team', 'home_score', 'away_score']].copy()
    home_df.columns = ['match_date', 'team', 'goals_scored', 'goals_conceded']
    away_df = df[['match_date', 'away_team', 'away_score', 'home_score']].copy()
    away_df.columns = ['match_date', 'team', 'goals_scored', 'goals_conceded']
    
    team_history = pd.concat([home_df, away_df]).sort_values(['team', 'match_date'])
    
    # 計算休息天數 (Rest Days)，最大值設為 30 天避免跨年季休數據異常
    team_history['rest_days'] = team_history.groupby('team')['match_date'].diff().dt.days
    team_history['rest_days'] = team_history['rest_days'].fillna(30).clip(upper=30)
    
    # 計算零封率 (Clean Sheet) 與近況
    team_history['is_clean_sheet'] = (team_history['goals_conceded'] == 0).astype(int)
    team_history['recent_score_avg'] = team_history.groupby('team')['goals_scored'].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    team_history['recent_concede_avg'] = team_history.groupby('team')['goals_conceded'].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    team_history['recent_clean_sheet_rate'] = team_history.groupby('team')['is_clean_sheet'].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    team_history = team_history.fillna(0)
    
    # 將所有特徵 JOIN 回主表
    join_cols = ['match_date', 'team', 'recent_score_avg', 'recent_concede_avg', 'recent_clean_sheet_rate', 'rest_days']
    
    df = pd.merge(df, team_history[join_cols], left_on=['match_date', 'home_team'], right_on=['match_date', 'team'], how='left')
    df = df.rename(columns={'recent_score_avg': 'home_recent_score', 'recent_concede_avg': 'home_recent_concede', 'recent_clean_sheet_rate': 'home_clean_sheet', 'rest_days': 'home_rest_days'}).drop(columns=['team'])
    
    df = pd.merge(df, team_history[join_cols], left_on=['match_date', 'away_team'], right_on=['match_date', 'team'], how='left')
    df = df.rename(columns={'recent_score_avg': 'away_recent_score', 'recent_concede_avg': 'away_recent_concede', 'recent_clean_sheet_rate': 'away_clean_sheet', 'rest_days': 'away_rest_days'}).drop(columns=['team'])

    # 計算休息天數優勢 (正值代表主隊休息較久)
    df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']

    logging.info(f"特徵工程 4.0 完成！產生了 {df.shape[1]} 個欄位。")
    return df

if __name__ == "__main__":
    df = create_features(get_matches_data())
    print("\n=== 預覽新特徵 (休息天數與防守率) ===")
    print(df[['match_date', 'home_team', 'away_team', 'home_rest_days', 'away_rest_days', 'home_clean_sheet']].tail())
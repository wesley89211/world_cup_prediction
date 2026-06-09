import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def compute_historical_elo(df, k_factor=32, default_elo=1500):
    # (ELO 計算邏輯保持不變，維持我們昨天的心血)
    elo_dict = {}
    home_elos, away_elos = [], []
    for idx, row in df.iterrows():
        h_team, a_team, h_res = row['home_team'], row['away_team'], row['home_result']
        h_elo, a_elo = elo_dict.get(h_team, default_elo), elo_dict.get(a_team, default_elo)
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        
        if h_res == 'W': w_h, w_a = 1.0, 0.0
        elif h_res == 'D': w_h, w_a = 0.5, 0.5
        else: w_h, w_a = 0.0, 1.0
            
        we_h = 1 / (10 ** (-(h_elo - a_elo) / 400) + 1)
        we_a = 1 - we_h
        elo_dict[h_team] = h_elo + k_factor * (w_h - we_h)
        elo_dict[a_team] = a_elo + k_factor * (w_a - we_a)
        
    df['home_elo'], df['away_elo'] = home_elos, away_elos
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df

def build_feature_matrix():
    logging.info("開始讀取 Processed 資料集...")
    data_dir = Path("data/processed")
    
    try:
        # 1. 讀取基準表並進行 ELO 計算
        results = pd.read_csv(data_dir / "cleaned_results.csv", low_memory=False)
        mapping = {'W': 2, 'D': 1, 'L': 0}
        results['result'] = results['home_result'].map(mapping)
        results = results.dropna(subset=['result'])
        results['is_neutral'] = results['neutral'].fillna(False).astype(int)
        
        # 先算完所有歷史的 ELO
        results = compute_historical_elo(results)

        # 🚨 關鍵時空過濾：算完 ELO 後，我們只取 2023 年以後的近期比賽來訓練
        # 這樣 2026 的身價特徵才不會產生時空錯亂
        if 'year' in results.columns:
            results = results[results['year'] >= 2023].copy()
            logging.info(f"已過濾 2023 年以後的賽事，剩餘 {len(results)} 場比賽進行終極訓練。")

        # 2. 讀取所有特徵表 (包含我們剛出爐的 Squad 特徵！)
        stats = pd.read_csv(data_dir / "wc26_team_stats.csv")
        form = pd.read_csv(data_dir / "wc26_form.csv")
        squads = pd.read_csv(data_dir / "wc26_squad_features.csv") # 👈 新武器進場
        stats = stats[stats['period'] == 'last_5_years'] 

        logging.info("開始進行全維度特徵整併 (JOIN)...")

        # --- 合併主隊特徵 ---
        df = results.merge(stats, left_on='home_team', right_on='team', how='left')
        df = df.rename(columns={'win_rate': 'home_win_rate', 'goals_for_avg': 'home_goals_for_avg', 'goals_against_avg': 'home_goals_against_avg', 'goal_diff_avg': 'home_goal_diff_avg'})
        df = df.merge(form, left_on='home_team', right_on='team', how='left')
        df = df.rename(columns={'form_score_official': 'home_form_score'})
        df = df.merge(squads, left_on='home_team', right_on='team', how='left')
        df = df.rename(columns={'total_value_eur': 'home_value', 'top_club_ratio': 'home_top_club_ratio'})
        df = df.drop(columns=['team_x', 'team_y', 'team', 'form_score_all', 'last_10_results', 'last_match_date'], errors='ignore')

        # --- 合併客隊特徵 ---
        df = df.merge(stats, left_on='away_team', right_on='team', how='left')
        df = df.rename(columns={'win_rate': 'away_win_rate', 'goals_for_avg': 'away_goals_for_avg', 'goals_against_avg': 'away_goals_against_avg', 'goal_diff_avg': 'away_goal_diff_avg'})
        df = df.merge(form, left_on='away_team', right_on='team', how='left')
        df = df.rename(columns={'form_score_official': 'away_form_score'})
        df = df.merge(squads, left_on='away_team', right_on='team', how='left')
        df = df.rename(columns={'total_value_eur': 'away_value', 'top_club_ratio': 'away_top_club_ratio'})
        df = df.drop(columns=['team_x', 'team_y', 'team', 'period', 'form_score_all', 'last_10_results', 'last_match_date'], errors='ignore')

        # --- 計算差距特徵 (The Difference Makers) ---
        logging.info("計算主客隊綜合實力差距 (包含資本身價差)...")
        df['win_rate_diff'] = df['home_win_rate'] - df['away_win_rate']
        df['goal_diff_avg_diff'] = df['home_goal_diff_avg'] - df['away_goal_diff_avg']
        df['form_score_diff'] = df['home_form_score'] - df['away_form_score']
        
        # 💰 終極武器：身價差與豪門佔比差！
        # 填補 0 是因為有些非決賽圈小國家沒有在我們的 800 人名單內
        df['home_value'] = df['home_value'].fillna(0)
        df['away_value'] = df['away_value'].fillna(0)
        df['home_top_club_ratio'] = df['home_top_club_ratio'].fillna(0)
        df['away_top_club_ratio'] = df['away_top_club_ratio'].fillna(0)
        
        # 為了避免數字過大，將歐元身價轉換為「百萬歐元」
        df['value_diff_m'] = (df['home_value'] - df['away_value']) / 1000000
        df['top_club_ratio_diff'] = df['home_top_club_ratio'] - df['away_top_club_ratio']

        df = df.fillna(0)

        y = df['result'] 
        
        # 🌟 最終全武裝特徵清單 🌟
        features = [
            'is_neutral',
            'home_elo', 'away_elo', 'elo_diff',  
            'home_win_rate', 'away_win_rate', 'win_rate_diff',
            'home_goals_for_avg', 'away_goals_for_avg',
            'home_goals_against_avg', 'away_goals_against_avg', 
            'home_goal_diff_avg', 'away_goal_diff_avg', 'goal_diff_avg_diff',
            'home_form_score', 'away_form_score', 'form_score_diff',
            'value_diff_m', 'top_club_ratio_diff' # 👈 星度情報正式上線！
        ]
        X = df[features]
        
        logging.info(f"特徵矩陣建置完成！共 {len(X)} 筆資料，已升級至 {len(features)} 個特徵。")
        return X, y

    except Exception as e:
        logging.error(f"資料整併發生錯誤：{e}")
        return None, None
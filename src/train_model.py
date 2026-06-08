import sqlite3
import pandas as pd
import logging
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DB_PATH = Path("data/world_cup.db")

def load_featured_data():
    from features import get_matches_data, create_features
    raw_df = get_matches_data()
    return create_features(raw_df)

def train_xgboost_model():
    logging.info("載入資料與特徵中...")
    df = load_featured_data()
    
    le_team = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']])
    le_team.fit(all_teams)
    
    df['home_team_encoded'] = le_team.transform(df['home_team'])
    df['away_team_encoded'] = le_team.transform(df['away_team'])
    
    features = [
        'home_team_encoded', 'away_team_encoded', 'tournament_weight', 'is_neutral',
        'home_recent_score', 'home_recent_concede', 'away_recent_score', 'away_recent_concede',
        'elo_diff', 'home_clean_sheet', 'away_clean_sheet', 'rest_days_diff'
    ]
    
    X = df[features]
    y = df['result']
    
    logging.info("切分資料集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 超參數最佳化 (Hyperparameter Tuning) ---
    logging.info("啟動超參數自動搜索 (這大約需要 1~3 分鐘，請耐心等候)...")
    
    # 定義要嘗試的參數範圍
    param_dist = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],         # 防止 Overfitting 的神參數
        'colsample_bytree': [0.8, 0.9, 1.0]   # 隨機抽取特徵比例
    }
    
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
    
    # 使用 RandomizedSearchCV 隨機測試 10 種組合 (n_iter=10 避免跑太久)
    random_search = RandomizedSearchCV(
        xgb_clf, param_distributions=param_dist, n_iter=15, 
        scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=1
    )
    
    random_search.fit(X_train, y_train)
    
    # 拿出最強的模型
    best_model = random_search.best_estimator_
    logging.info(f"最佳參數組合: {random_search.best_params_}")
    
    # --- 模型評估 ---
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("🏆 XGBoost (最佳化版) 成果報告 🏆")
    print("="*40)
    print(f"整體預測準確率 (Accuracy): {accuracy:.2%}")
    print("-" * 40)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*40)
    
    # 將預測結果寫回資料庫
    logging.info("準備將預測結果寫回 SQLite 資料庫...")
    X_all = df[features]
    probabilities = best_model.predict_proba(X_all)
    
    predictions_df = pd.DataFrame({
        'match_id': df['match_id'],
        'predicted_result': best_model.predict(X_all),
        'prob_away_win': probabilities[:, 0],
        'prob_draw': probabilities[:, 1],
        'prob_home_win': probabilities[:, 2]
    })
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            predictions_df.to_sql('predictions', conn, if_exists='replace', index=False)
            logging.info("成功建立 predictions 資料表並寫入預測數據！")
    except Exception as e:
        logging.error(f"預測數據寫入失敗：{e}")

if __name__ == "__main__":
    train_xgboost_model()
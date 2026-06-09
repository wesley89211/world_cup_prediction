import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_pipeline():
    logging.info("啟動預測模型訓練管線...")
    
    # 1. 取得特徵與標籤
    X, y = build_feature_matrix()
    if X is None:
        return

    # 2. 切割訓練集與測試集 (80% 訓練, 20% 測試)
    # random_state=42 確保每次切出來的結果一樣，方便比較準確率
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 初始化 XGBoost 模型
    logging.info("開始訓練 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=1  # 避免 Windows 環境報錯
    )

    # 4. 訓練模型
    model.fit(X_train, y_train)
    logging.info("模型訓練完成！")

    # 5. 進行預測與評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("🏆 XGBoost 基準線模型評估報告 🏆")
    print("="*40)
    print(f"✅ 整體準確率 (Accuracy): {acc:.4f} ({(acc*100):.2f}%)")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['客勝(0)', '和局(1)', '主勝(2)']))
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
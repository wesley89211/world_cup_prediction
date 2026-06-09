# 📝 工作日誌 (Daily Log) - 2026-06-08

專案：世界盃賽事預測系統 (World Cup Prediction Pipeline)

--1. 特徵工程與模型迭代 (Feature Engineering & Modeling)
完成特徵工程 2.0 (近期狀態)： 利用 Pandas .shift().rolling() 實作主客隊近 5 場得失球移動平均，將模型整體準確率提升至 53.95%，大幅改善了客場勝率 (Class 0) 的召回率。

完成特徵工程 3.0 (動態 ELO 積分)： 捨棄外部靜態 ELO 檔案，直接在 Python 內建動態計算歷史 ELO 積分的演算法（包含賽事權重調整），準確率突破至 58.14%。

完成特徵工程 4.0 (隱藏特徵挖掘)： 利用現有時間與主客場資料，衍生計算出「休息天數差距 (Rest Days)」與「零封率 (Clean Sheet Rate)」，使和局預測的精準度 (Precision) 有顯著提升。

實作超參數最佳化 (Hyperparameter Tuning)： 導入 RandomizedSearchCV 進行 XGBoost 參數自動搜索。期間排除並修復了 Windows 環境下多核心運算與中文路徑衝突的 Bug (設定 n_jobs=1)，最佳化後準確率穩定在 58.09%。

完成預測結果寫回機制： 成功將模型預測結果與三種賽果的機率 (predict_proba)，自動寫回 SQLite 資料庫的 predictions 資料表中。

--2. 資料獲取管線開發 (Data Ingestion Pipeline)
開發維基百科名單爬蟲： 針對 2026 年世界盃大名單 (English Wikipedia) 進行爬蟲開發。

排除 DOM 解析障礙： 實作過程中遇到網頁表格尚未完全實體化以及隱藏容器 (div) 的結構陷阱。

前端注入爬取與清洗： 轉為利用 Client-side JavaScript 在瀏覽器端擷取 table.wikitable，產出初步 JSON 資料。接著利用 Python 與正則表達式 (re.sub) 清除多餘註腳標記。

原始資料落地： 成功抓取並彙整出包含球員姓名、年齡、出場數、進球數與效力俱樂部的名單，輸出為 wc2026_squads.csv。

--3. 專案架構與版本控制 (Architecture & Version Control)
檔案歸檔： 將原始資料 (wc2026_squads.csv) 配置於 data/raw/ 目錄，將爬蟲腳本 (scrape_wc2026_squads.py) 配置於 src/ 目錄。

執行 Git 提交： 驗證 .gitignore 成功阻擋原始資料庫檔案上傳，並成功將爬蟲程式碼 Push 至 GitHub，Commit 訊息為 feat: 新增 2026 世界盃大名單爬蟲程式。
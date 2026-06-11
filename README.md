# 2026 FIFA World Cup 預測系統

用機器學習預測 48 支國家隊的分組賽結果，每場比賽輸入真實比分後自動更新後續預測。

---

## 快速開始

```bash
# 1. 產出全部 72 場預測（只需執行一次）
python src/verify_predictions.py --predict

# 2. 早上查看今天有哪些比賽
python src/verify_predictions.py --today

# 3. 賽後輸入比分（自動更新後續預測）
python src/verify_predictions.py --verify

# 4. 查看積分榜
python src/verify_predictions.py --standings

# 5. 查看模型準確率
python src/verify_predictions.py --stats
```

---

## 指令說明

| 指令 | 說明 |
|---|---|
| `--predict` | 產出所有預測，用最新 ELO 和狀態重算 |
| `--today` | 查看今天台灣時間的比賽與預測 |
| `--verify` | 輸入比分，自動更新狀態並重算後續比賽 |
| `--standings` | 查看目前積分榜 |
| `--stats` | 查看累計預測準確率 |

搭配 `--date YYYY-MM-DD` 指定台灣日期，`--group A` 指定組別（standings 用）。

---

## 每日流程

```
早上  →  --today    查看今天幾點有哪些比賽
賽後  →  --verify   輸入比分（Enter 可跳過未結束的場次）
隨時  →  --standings / --stats
```

輸入比分後系統會自動：
1. 更新雙方 ELO（WC 用 K=64，爆冷影響更大）
2. 更新近期 form 分數
3. 更新積分榜
4. 重算所有後續比賽的預測

---

## 輸出說明

**終端機顯示範例：**
```
台灣時間 2026-06-14 06:00  │  Group C
Brazil                vs  Morocco
ELO：Brazil 1953  vs  Morocco 1918
主場勝  46.9%  │  平局  24.9%  │  客場勝  28.2%
預測結果 ▶ Brazil
比分推薦 ▶ #1 2-1(8.3%)  #2 3-1(8.2%)  #3 2-0(7.7%)
```

**預測記錄：** `data/processed/wc2026_predictions.csv`
**滾動狀態：** `data/processed/wc2026_live_state.csv`

---

## 模型簡介

| 項目 | 說明 |
|---|---|
| 訓練資料 | 歷史國際賽事 49,365 場（1872~2026） |
| 球員資料 | Wikipedia 大名單 + Transfermarkt 身價 + EA FC 25 能力值 |
| 核心模型 | XGBoost（30%）+ 雙 Poisson 回歸（70%）Ensemble |
| 測試準確率 | 59.58%（最後 20% 歷史賽事） |
| 最重要特徵 | ELO 分差（37%）、絕對 ELO（40%）、EA 整體戰力差（14%） |

---

## 檔案結構

```
src/
├── verify_predictions.py      每天用這個
├── build_match_dataset_v2.py  建立訓練資料
├── train_advanced_v2.py       訓練進階模型
├── train_baseline.py          訓練基準模型
├── build_final_features.py    合併特徵表
└── wc2026_report.py           48 隊多主題文字報告

data/
├── models/    訓練好的模型 (.pkl)
├── processed/ 特徵表、預測記錄、滾動狀態
└── raw/       原始資料
```

---

## 環境需求

```bash
pip install pandas numpy scikit-learn xgboost scipy
```

Python 3.11，Windows 測試通過。

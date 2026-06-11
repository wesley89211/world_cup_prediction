"""
test_bsd_results.py
確認 BSD API 賽果格式
"""
import requests, json

API_KEY = "6c8388e0a102b8f1c3e160ddda55310b802dd003"
HEADERS = {"Authorization": f"Token {API_KEY}"}
BASE    = "https://sports.bzzoiro.com"

# 拉 WC 全部比賽，看已結束的比分格式
r = requests.get(f"{BASE}/api/events/", headers=HEADERS,
                 params={"league": 27, "limit": 20})
data = r.json()

print(f"共 {data['count']} 場，顯示前幾場：\n")
for e in data["results"]:
    status     = e.get("status","")
    home_score = e.get("home_score")
    away_score = e.get("away_score")
    ht_h       = e.get("home_score_ht")
    ht_a       = e.get("away_score_ht")
    minute     = e.get("current_minute")
    period     = e.get("period","")

    print(f"  [{e['id']}] {e['home_team']} vs {e['away_team']}")
    print(f"    status={status}  score={home_score}-{away_score}"
          f"  HT={ht_h}-{ht_a}  minute={minute}  period={period}")
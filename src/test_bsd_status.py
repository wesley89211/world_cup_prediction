"""確認剩下比賽的 BSD event id"""
import requests
from datetime import datetime, timezone, timedelta

HEADERS = {"Authorization": "Token 6c8388e0a102b8f1c3e160ddda55310b802dd003"}

# 從列表拿所有未開賽的比賽
r = requests.get("https://sports.bzzoiro.com/api/events/",
                 headers=HEADERS, params={"league": 27, "limit": 100})
events = r.json().get("results", [])

NAME_MAP = {
    "Czechia":"Czech Republic","Türkiye":"Turkey",
    "Côte d'Ivoire":"Ivory Coast","Cabo Verde":"Cape Verde",
    "USA":"United States","Bosnia & Herzegovina":"Bosnia and Herzegovina",
}

print(f"# BSD event id 對照表（共 {len(events)} 場）")
print('BSD_EVENT_IDS = {')
for e in sorted(events, key=lambda x: x["id"]):
    h = NAME_MAP.get(e["home_team"], e["home_team"])
    a = NAME_MAP.get(e["away_team"], e["away_team"])
    dt = datetime.fromisoformat(e["event_date"].replace("Z","+00:00"))
    tw = dt.astimezone(timezone(timedelta(hours=8)))
    print(f'    ("{h:<25}", "{a:<25}"): {e["id"]},  # {tw.strftime("%m/%d %H:%M")}')
print('}')

# 也直接確認 Mexico vs SA
r2 = requests.get("https://sports.bzzoiro.com/api/events/8287/", headers=HEADERS)
e2 = r2.json()
print(f"\n# 開幕戰確認：id=8287  {e2['home_team']} {e2['home_score']}-{e2['away_score']} {e2['away_team']}")
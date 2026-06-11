"""
test_bsd_player_stats.py
========================
確認 BSD API 對世界盃球員的統計數據覆蓋率
執行：python src/test_bsd_player_stats.py
"""
import requests
import json

API_KEY = "6c8388e0a102b8f1c3e160ddda55310b802dd003"
BASE    = "https://sports.bzzoiro.com"
HEADERS = {"Authorization": f"Token {API_KEY}"}
SEP = "─" * 60

def get(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params or {}, timeout=15)
    print(f"  GET {path} → {r.status_code}")
    if r.status_code == 200:
        return r.json()
    print(f"  ❌ {r.text[:200]}")
    return {}

# ── 1. 搜尋世界盃明星球員，確認有沒有資料 ────────────────────
print(f"\n{SEP}")
print("1. 搜尋代表性球員")
print(SEP)

test_players = ["Vinicius", "Mbappe", "Haaland", "Bellingham", "Yamal"]
for name in test_players:
    data = get("/api/players/", {"search": name, "limit": 2})
    results = data.get("results", [])
    if results:
        p = results[0]
        print(f"  {name} → [{p['id']}] {p['name']} | club={p.get('club_name',p.get('team','?'))} | pos={p.get('position','?')}")
    else:
        print(f"  {name} → 找不到")

# ── 2. 查一位球員的詳細統計 ──────────────────────────────────
print(f"\n{SEP}")
print("2. 查詢球員詳細統計（先找 Vinicius）")
print(SEP)

data = get("/api/players/", {"search": "Vinicius Junior", "limit": 3})
for p in data.get("results", []):
    print(f"  [{p['id']}] {p['name']} | 欄位：{list(p.keys())}")

if data.get("results"):
    pid = data["results"][0]["id"]
    detail = get(f"/api/players/{pid}/")
    print(f"\n  完整欄位：{list(detail.keys())[:20]}")

# ── 3. 查球員賽季統計 endpoint ────────────────────────────────
print(f"\n{SEP}")
print("3. 球員統計 endpoint 測試")
print(SEP)

# 試幾個可能的 endpoint
for path in ["/api/player-stats/", "/api/playerstats/", "/api/players/stats/"]:
    data = get(path, {"limit": 1})
    if data and "results" in data:
        print(f"  ✅ {path} 有資料，欄位：{list(data['results'][0].keys())[:15] if data['results'] else '空'}")
        break

# ── 4. 查一場 WC 比賽的球員統計 ──────────────────────────────
print(f"\n{SEP}")
print("4. 從 WC 比賽查球員統計（event_id=8287 = Mexico vs South Africa）")
print(SEP)

# 先試 v1
for path in ["/api/events/8287/", "/api/events/8287/player-stats/", "/api/events/8287/lineups/"]:
    data = get(path)
    if data:
        keys = list(data.keys()) if isinstance(data, dict) else "list"
        print(f"  {path} → keys={keys[:10] if isinstance(keys, list) else keys}")
        if isinstance(data, dict) and "lineups" in data:
            print(f"    lineups: {json.dumps(data['lineups'], ensure_ascii=False)[:300]}")
        if isinstance(data, dict) and "player_stats" in data:
            print(f"    player_stats: {json.dumps(data['player_stats'], ensure_ascii=False)[:300]}")

# ── 5. 試 v2 endpoint ─────────────────────────────────────────
print(f"\n{SEP}")
print("5. BSD v2 endpoint 測試")
print(SEP)

for path in ["/api/v2/events/8287/", "/api/v2/player-stats/?event=8287"]:
    data = get(path)
    if data:
        keys = list(data.keys()) if isinstance(data, dict) else "list"
        print(f"  {path} → keys={keys[:10] if isinstance(keys, list) else keys}")

# ── 6. 查 /api/players/ 有哪些統計欄位 ──────────────────────
print(f"\n{SEP}")
print("6. /api/players/ 全欄位（取一筆看有哪些統計）")
print(SEP)

data = get("/api/players/", {"limit": 1})
if data.get("results"):
    p = data["results"][0]
    print(json.dumps(p, indent=2, ensure_ascii=False)[:1500])

print(f"\n{SEP}")
print("完成！把輸出貼給 Claude")
print(SEP)

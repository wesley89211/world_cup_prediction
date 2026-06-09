import pandas as pd
import numpy as np

def create_squad_features():
    print("📥 開始讀取 2026 世界盃大名單與身價資料...")
    
    try:
        df = pd.read_csv("data/processed/wc2026_squads_final.csv")
    except FileNotFoundError:
        print("❌ 找不到檔案！請確認資料夾路徑。")
        return

    # 🎯 根據終端機輸出的真實欄位，直接進行精準綁定
    team_col = 'Country'
    club_col = 'Club'
    age_col = 'Age'
    caps_col = 'Caps'
    goals_col = 'Goals'
    pos_col = 'Pos.'                # 解決帶有縮寫點的問題
    value_col = 'market_value_eur'  # 解決身價欄位名稱不同的問題
    
    print(f"✅ 成功鎖定所有欄位！準備進行特徵聚合...")

    # 1. 基礎聚合：總身價、平均年齡、總出場數、總進球數
    squad_stats = df.groupby(team_col).agg(
        total_value_eur=(value_col, 'sum'),
        avg_age=(age_col, 'mean'),
        total_caps=(caps_col, 'sum'),
        total_goals=(goals_col, 'sum')
    ).reset_index()

    # 2. 進階特徵 A：豪門球員比例 (Top Club Ratio)
    top_clubs = ['Real Madrid', 'Barcelona', 'Bayern', 'Dortmund', 'Leverkusen',
                 'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Man Utd', 'Tottenham',
                 'Inter', 'Milan', 'Juventus', 'Napoli', 'PSG', 'Atletico']
    
    df['is_top_club'] = df[club_col].str.contains('|'.join(top_clubs), case=False, na=False).astype(int)
    top_club_ratio = df.groupby(team_col)['is_top_club'].mean().reset_index().rename(columns={'is_top_club': 'top_club_ratio'})

    # 3. 進階特徵 B：進攻火力佔比 (頭重腳輕指數)
    df['is_forward'] = df[pos_col].str.contains('Forward|Striker|Winger|FW', case=False, na=False)
    forward_value = df[df['is_forward']].groupby(team_col)[value_col].sum().reset_index().rename(columns={value_col: 'forward_value'})
    
    # 合併所有特徵
    final_features = squad_stats.merge(top_club_ratio, on=team_col, how='left')
    final_features = final_features.merge(forward_value, on=team_col, how='left')
    
    # 處理比例與缺失值
    final_features['forward_value'] = final_features['forward_value'].fillna(0)
    final_features['attack_value_ratio'] = np.where(
        final_features['total_value_eur'] > 0, 
        final_features['forward_value'] / final_features['total_value_eur'], 
        0
    )
    final_features = final_features.drop(columns=['forward_value'])

    # 🚨 為了讓 XGBoost 能無縫接軌，把 'Country' 強制改名為 'team'
    final_features = final_features.rename(columns={team_col: 'team'})

    # 💾 儲存為模型可讀的高級特徵檔
    output_path = "data/processed/wc26_squad_features.csv"
    final_features.to_csv(output_path, index=False)
    
    print(f"\n✅ 成功產出國家隊星度特徵！檔案已儲存至：{output_path}")
    print("-" * 40)
    print("💰 2026 世界盃總身價 Top 5 國家隊預覽：")
    
    top_5 = final_features.sort_values('total_value_eur', ascending=False).head(5)
    for idx, row in top_5.iterrows():
        value_in_m = row['total_value_eur'] / 1000000  # 轉換為百萬歐元
        print(f"⚽ {row['team']:<15} | {value_in_m:>6.1f} 百萬歐元 | 豪門佔比: {row['top_club_ratio']*100:>4.1f}% | 平均年齡: {row['avg_age']:.1f}")

if __name__ == "__main__":
    create_squad_features()
import pandas as pd

# 被験者のIDリスト
ID_list = [f'ID{i:03d}' for i in range(20)] + [f'ID{i:03d}' for i in range(101, 106)]

# 各被験者のデータを格納するリスト
data_frames = []

# 各被験者のCSVファイルを読み込んでリストに追加
for i in ID_list:
    # CSVファイルをロード
    df = pd.read_csv(f"./AOI/{i}_AOI.csv")

    # 値が0の場合をNaNに変換
    df.replace(0, pd.NA, inplace=True)
    
    # 平均値を計算して新しい行として追加
    df_mean = df.mean().to_frame().transpose()
    df_mean.insert(0, 'ID', i[2:])  # IDを一番左に挿入
    
    # リストに追加
    data_frames.append(df_mean)

# データフレームを作成
df_result = pd.concat(data_frames, ignore_index=True)

# NaNを0に変換
df_result = df_result.fillna(0)

# 二つのCSVファイルを読み込む
df1 = pd.read_csv('kadai1.csv', dtype={'ID': str})
df2 = df_result

# df2から 'ID' 列を除外
df2_without_id = df2.drop(columns=['ID'])

# データフレームを横に結合（横に並べる）
result = pd.concat([df1, df2_without_id], axis=1)

result.to_csv("results_example.csv", index=False)
result
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def getKeyResponse(GazeDataFrame):
    tmp = GazeDataFrame[GazeDataFrame['Event'].isin(['KeyboardEvent'])]
    Response = tmp[1:16][tmp['Event value'] != 'space']['Event value'].values
    return Response

tmp = pd.read_table("./Class_EyeTracking/CorrectAnswer.csv")
Correct = tmp['Correct Answer'].values

ID_list = [f'ID{i:03d}' for i in range(20)] + [f'ID{i:03d}' for i in range(101, 106)]

result_dfs = []  # 各IDごとの結果を格納するためのリスト

for i in ID_list:
    # csvファイルをロード
    df = pd.read_csv(f"./Class_EyeTracking/data/{i}_DataExport.csv")
    
    # getKeyResponse関数で 'Response' 列を生成
    Response = getKeyResponse(df)

    # Hit, False Alarm, Miss, Correct Rejectionの計算
    TP = sum((Response == 'Right') & (Correct == 'Right'))
    FP = sum((Response == 'Right') & (Correct == 'Left'))
    FN = sum((Response == 'Left') & (Correct == 'Right'))
    TN = sum((Response == 'Left') & (Correct == 'Left'))

    # 評価指標の計算
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F = (2 * Precision * Recall) / (Precision + Recall)
    KeyMiss = 15 - TP - FP - FN - TN

    # 結果のDataFrameを作成
    result_df = pd.DataFrame({
        'ID': i[2:],
        'Accuracy': [Accuracy],
        'Precision': [Precision],
        'Recall': [Recall],
        "F-measure": [F],
        'TP': [TP],
        'FN': [FN],
        'FP': [FP],
        'TN': [TN],
        'KeyMiss': [KeyMiss]
    })

    # 結果をリストに追加
    result_dfs.append(result_df)

# 全ての結果を1つのDataFrameに結合
final_result_df = pd.concat(result_dfs, ignore_index=True)
final_result_df.to_csv("kadai1.csv", index=False)

# 結果のDataFrameを表示
final_result_df
# 課題3：被験者1名のAOI時間を、画像刺激間で平均する。
import os
import pandas as pd
import numpy as np

ID_list = [f'ID{i:03d}' for i in range(20)] + [f'ID{i:03d}' for i in range(101, 106)]

for ID in ID_list:
    ID = ID
    raw = pd.read_table('./Class_EyeTracking/Metrics/' + ID + '_Metrics.tsv')

    cutLabels = []  # 抽出するデータラベルを保存するリスト
    for label in raw.columns:
        # if文で特定の文字列を含む列を抽出
        if label.find('Total_duration_of_fixations') \
            != -1 or label.find('Duration_of_interval') \
                != -1 or label.find('TOI') != -1:
            cutLabels.append(label)  # リストに追加

    # データを抽出
    df = raw[cutLabels]

    NUM = 15  # 呈示刺激の数; 今回は 15なので 15+1=16
    TOI = raw['TOI'].iloc[0:NUM]  # TOIを初期化; rawデータから引っ張ってくる
    res = np.zeros([NUM, 4])

    # 'Total_duration_of_fixations'が含まれる列を抽出
    total_fixations_columns = [col for col in df.columns \
                            if 'Total_duration_of_fixations' in col]

    # 各列の合計を計算
    total_fixations = df[total_fixations_columns].sum(axis=1)

    # TOIでloop 処理
    for i in range(NUM):
        # 計算用変数の初期化
        Header = 0
        HeaderTime = 0
        Footer = 0
        URL = 0
        
        # 欠損値（NaN）を含むデータラベルを除外
        tmp = df[i:i + 1].dropna(how="all", axis=1) 
        # 'Total_duration_of_fixations'列の値をTOTALに変更
        #TOTAL = np.array(total_fixations.iloc[i])  
        TOTAL = np.array(tmp['Duration_of_interval'].values)

        # データラベルで loop 処理
        for label in tmp.columns:
            # データラベルが Header で終わる列を抽出
            if label.endswith('Header') == 1:
                Header += tmp[label].values
            # データラベルが HeaderTime で終わる列を抽出
            if label.endswith('HeaderTime') == 1:
                HeaderTime += tmp[label].values
            # データラベルが Footer で終わる列を抽出
            if label.endswith('Footer'):
                Footer += tmp[label].values
            # データラベルに URL を含む列を抽出
            if label.find('URL') != -1:
                URL += tmp[label].values

        # 結果を格納
        Header = np.array(Header)
        HeaderTime = np.array(HeaderTime)
        Footer = np.array(Footer)
        URL = np.array(URL)

        # 各変数を正規化してから結合
        calc = np.column_stack([
                Header, HeaderTime, Footer, URL]) / TOTAL[:, np.newaxis]
        res[i, :] = calc[0]

    # DataFrame として統合
    df_sum = pd.DataFrame(res,
                        columns=['Header', 'HeaderTime', 'Footer', 'URL'],
                        index=TOI)
    # 出力ディレクトリの指定
    output_directory = 'AOI/'

    # ファイル名の生成
    csv_filename = f"{ID}_AOI.csv"

    # 完全なファイルパスの生成
    output_filepath = os.path.join(output_directory, csv_filename)
    os.makedirs(output_directory, exist_ok=True)

    # CSVファイルに出力
    df_sum.to_csv(output_filepath, index=False)
    
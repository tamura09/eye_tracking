# ライブラリインポート
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

ID_list = [f'ID{i:03d}' for i in range(20)] + [f'ID{i:03d}' for i in range(101, 106)]
stimuli_list = ['01_Kyoumu_T.png', '02_Amazon2_F.png', '03_Amazon3_F.png',
                '04_Rakuten2_F.png', '05_ticket_T.png', '06_Rakuten_F.png',
                '07_Rakuten_T.png', '08_yodobasi_F.png', '09_Apple1_F.png',
                '10_Kankou_T.png', '11_LINE_F.png', '12_Kyufu2_F.png',
                '13_ponta_T.png', '14_Kyufu_F.png', '15_SMBC_F.png']
result_data = {'ID': ID_list}

def getTOI(GazeDataFrame, MediaName):
    TOI = GazeDataFrame[GazeDataFrame['Presented Media name'].isin([MediaName])]
    return TOI

for MediaName in stimuli_list:
    durations = []
    for ID in ID_list:
        df = pd.read_csv(f"./Class_EyeTracking/data/{ID}_DataExport.csv")  # Experiment data path
        TOI = getTOI(df, MediaName)

        if not TOI.empty:
            TOI_start = TOI['Recording timestamp'].iloc[1]
            TOI_end = TOI['Recording timestamp'].iloc[-1]
            Duration = (TOI_end - TOI_start) / 1000000
            durations.append(Duration)
        else:
            durations.append(None)
    result_data[MediaName] = durations

# Convert the dictionary to a DataFrame
result_df = pd.DataFrame(result_data)

# 各IDの平均持続時間を計算
result_df['Mean Duration'] = result_df.iloc[:, 1:].mean(axis=1)

# 最大および最小の平均持続時間を持つIDを見つける
max_mean_duration_ID = result_df.loc[result_df['Mean Duration'].idxmax()]['ID']
min_mean_duration_ID = result_df.loc[result_df['Mean Duration'].idxmin()]['ID']
max_mean_duration = result_df['Mean Duration'].max()
min_mean_duration = result_df['Mean Duration'].min()

# 結果を表示
print("平均TOI持続時間が最大のID:")
print("ID:", max_mean_duration_ID, "- 平均持続時間:", max_mean_duration)
print("\n平均TOI持続時間が最小のID:")
print("ID:", min_mean_duration_ID, "- 平均持続時間:", min_mean_duration)

duration_list = ['min', 'max']

for duration_type in duration_list:
    for MediaName in stimuli_list:
        #----- 被験者応答読み込み
        ID = max_mean_duration_ID if duration_type == 'max' else min_mean_duration_ID

        # csvファイルをロード
        df = pd.read_csv("./Class_EyeTracking/data/" + ID + "_DataExport.csv")

        # TOI抽出
        TOI = getTOI(df, MediaName)

        # 視線位置の X, Y を取得。この値を使ってプロットする。
        GazeX = TOI['Gaze point X']
        GazeY = TOI['Gaze point Y']

        # 刺激画像の読み込み;
        img_bgr = cv2.imread(
                    './Class_EyeTracking/stimuli/' + MediaName[:-4] + '_Stim.png')
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 画像の高さと幅を取得
        img_height, img_width, _ = img.shape

        # 画像内に収まるように視線位置をクリップ
        GazeX = GazeX.clip(0, img_width - 1)
        GazeY = GazeY.clip(0, img_height - 1)

        # 画像の表示
        plt.figure()
        plt.imshow(img)
        plt.plot(GazeX, GazeY, 'r.-')
        # タイトルを追加
        plt.title("Line of sight for {} - ID: {}".format(MediaName, ID[2:]))

        png_output_folder = f'./課題2/{ID}/gaze/png/'
        os.makedirs(png_output_folder, exist_ok=True)
        filename = png_output_folder + ID + '_gaze_' + MediaName
        plt.savefig(filename[:-4] + '.png')

        svg_output_folder = f'./課題2/{ID}/gaze/svg/'
        os.makedirs(svg_output_folder, exist_ok=True)
        filename = svg_output_folder + ID + '_gaze_' + MediaName
        plt.savefig(filename[:-4] + '.svg')

        eps_output_folder = f'./課題2/{ID}/gaze/eps/'
        os.makedirs(eps_output_folder, exist_ok=True)
        filename = eps_output_folder + ID + '_gaze_' + MediaName
        plt.savefig(filename[:-4] + '.eps')
        plt.close()

        # 1. ----- 画像配列初期化
        imSize = img.shape
        heatmap = np.zeros([imSize[0], imSize[1]], dtype='float32')

        # イメージのサイズを取得
        image_height, image_width = imSize[:2]

        # 2.～4. ----- 視線位置が記録された画素をインクリメント
        for i in range(len(GazeX)):
            # 視線位置が記録されていれば処理（Eyes Not Found の場合もあるので。）
            if not (np.isnan(GazeX.iloc[i]) or np.isnan(GazeY.iloc[i])):
                # インデックスが画像サイズを超えないように制約
                xx = min(int(GazeX.iloc[i]), image_width - 1)
                yy = min(int(GazeY.iloc[i]), image_height - 1)
                heatmap[yy, xx] = heatmap[yy, xx] + 1

        # 5. ----- ぼかしを適用
        # 標準偏差=50pixのガウス分布を使用
        # ガウス分布が0に漸近するのに十分なカーネルサイズを指定; 標準偏差の4倍あればOK
        heatmap_blur = cv2.GaussianBlur(heatmap, (201,201), 50)
        # 画像を 0-1 に正規化
        MAX = np.max(heatmap_blur)  # 最大値
        MIN = np.min(heatmap_blur)  # 最小値
        heatmap_blur = (heatmap_blur - MIN)/(MAX - MIN) # 正規化

        #----- 表示するための作業
        # ヒートマップをRGBに変換
        heatmap_rgb = np.uint8(255 * heatmap_blur)
        # JETのカラーマップを適用
        heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        # RGBに変換
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        superimposed_img = heatmap_rgb*0.4 + img
        superimposed_img = superimposed_img/np.max(superimposed_img)

        # 画像を表示
        fig = plt.figure()
        # タイトルを追加
        plt.title("Heatmap for {} - ID: {}".format(MediaName, ID[2:]))
        plt.imshow(superimposed_img)

        # 保存
        png_output_folder = f'./課題2/{ID}/heatmap/png/'
        os.makedirs(png_output_folder, exist_ok=True)
        filename = png_output_folder + ID + '_heatmap_' + MediaName
        plt.savefig(filename[:-4] + '.png')

        svg_output_folder = f'./課題2/{ID}/heatmap/svg/'
        os.makedirs(svg_output_folder, exist_ok=True)
        filename = svg_output_folder + ID + '_heatmap_' + MediaName
        plt.savefig(filename[:-4] + '.svg')

        eps_output_folder = f'./課題2/{ID}/heatmap/eps/'
        os.makedirs(eps_output_folder, exist_ok=True)
        filename = eps_output_folder + ID + '_heatmap_' + MediaName
        plt.savefig(filename[:-4] + '.eps')
        plt.close()

result_df
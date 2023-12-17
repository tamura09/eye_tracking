import os
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータフレームを読み込む
df = pd.read_csv('results_example.csv')

# 横軸を F値、縦軸をAOIにしてプロット
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# 散布図を作成
df.plot.scatter('F-measure', 'Header', ax=axes[0, 0])
df.plot.scatter('F-measure', 'HeaderTime', ax=axes[0, 1])
df.plot.scatter('F-measure', 'Footer', ax=axes[1, 0])
df.plot.scatter('F-measure', 'URL', ax=axes[1, 1])

# 相関係数の算出
correlation_matrix = df.corr()

# グラフを保存
output_folder = './課題4/'
os.makedirs(output_folder, exist_ok=True)
fig.savefig(os.path.join(output_folder, 'scatter_plots.svg'))
fig.savefig(os.path.join(output_folder, 'scatter_plots.png'))

# 相関係数をCSVファイルとして保存
correlation_matrix.to_csv(os.path.join(output_folder, 'correlation_matrix.csv'), index=False)


# 表示
plt.show()

# 相関係数を表示
df.corr()

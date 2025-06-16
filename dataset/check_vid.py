import os
import csv

# 设定CSV文件路径
part_name = 'part15'
csv_path = f'./OpenVidHD_{part_name}/{part_name}_videos.csv'

# 1. 获取当前目录下所有 .mp4 文件的文件名
mp4_files = [f for f in os.listdir(f'./OpenVidHD_{part_name}/') if f.lower().endswith('.mp4')]
mp4_set = set(mp4_files)

# 2. 读取 CSV 中的 'Filename' 列
csv_filenames = set()
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_filenames.add(row['Filename'].strip())

# 3. 找出未出现在 CSV 中的 MP4 文件
missing_in_csv = mp4_set - csv_filenames

# 4. 输出结果
if missing_in_csv:
    print("以下 MP4 文件未出现在 CSV 文件中：")
    for filename in sorted(missing_in_csv):
        print(f" - {filename}")
else:
    print("所有 MP4 文件都已包含在 CSV 文件中。")

# 5. 找出出现在 CSV 中但文件夹中不存在的 MP4 文件
missing_mp4 = csv_filenames - mp4_set
if missing_mp4:
    print("\n以下文件在 CSV 中存在，但实际文件不存在：")
    for filename in sorted(missing_mp4):
        print(f" - {filename}")

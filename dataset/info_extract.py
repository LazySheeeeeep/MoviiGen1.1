import json
import csv

# 输入路径
json_path = '/dataset-v2/yy/MoviiGen1.1/OpenVid-1M/OpenVidHD/OpenVidHD.json'
full_csv_path = '/dataset-v2/yy/MoviiGen1.1/OpenVid-1M/data/train/OpenVidHD.csv'  # 原始完整csv文件
part_name = 'part15'               # 你下载的part名字
output_csv_path = f'./OpenVidHD_{part_name}/{part_name}_videos.csv'

# 1. 读取JSON文件，获取part中所有视频文件名
with open(json_path, 'r', encoding='utf-8') as f:
    parts = json.load(f)

video_list = []
for entry in parts:
    if part_name in entry:
        video_list = entry[part_name]
        break

if not video_list:
    raise ValueError(f"未找到指定的 part：{part_name}")

video_set = set(video_list)

# 2. 读取总CSV文件，提取匹配项
rows_to_write = []
with open(full_csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['video'] in video_set:
            rows_to_write.append({
                'Filename': row['video'],
                'Video Description': row['caption']
            })

# 3. 写入新的CSV文件
with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Filename', 'Video Description'])
    writer.writeheader()
    writer.writerows(rows_to_write)

print(f"成功写入 {len(rows_to_write)} 条记录到 {output_csv_path}")

import json

input_path = './OpenVidHD_part15_latents/videos2caption_latest.json'         # 原始 JSON 文件路径
output_path = './OpenVidHD_part15_latents/videos2caption_latest_test.json'   # 截取前1000项后的输出文件

# 读取原始 JSON 文件
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 确保是一个列表
if not isinstance(data, list):
    raise ValueError("JSON 文件的顶层结构必须是列表")

# 截取前1000条
subset = data[:1000]

# 写入新 JSON 文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print(f"✅ 成功提取前 {len(subset)} 条数据到 {output_path}")

#提取像素代码
import csv
import os

# 定义文件名的起始值和结束值
#start = 1223
#end = 3222
start = 3223
end = 5222
file_path_prefix = 'rawdata/'  # 文件路径前缀

# 创建一个空的数据列表
data = []

# 循环读取文件并将数据添加到列表中
for file_number in range(start, end + 1):
    file_name = f'{file_path_prefix}{file_number}'
    try:
        with open(file_name, 'rb') as file:
            # 读取文件内容
            content = file.read()

            # 将每个数字添加到数据列表中
            row_data = list(content)
            data.append(row_data)
    except FileNotFoundError:
        print(f"文件 '{file_name}' 不存在，跳过处理。")

# 保存数据为CSV文件
with open('DSdata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入CSV文件
    writer.writerows(data)

print('任务完成.')

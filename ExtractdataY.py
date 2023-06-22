#提取性别标签代码
import csv
# 打开文本文件并读取内容
with open('faceDS.txt', 'r') as file:  #or faceDR.txt
    content = file.read()
# 提取和转换_sex字段的值
lines = content.split('\n')  # 按行分割文本
data = []
for line in lines:
    if '(_sex' in line:
        # 提取_sex字段的值
        sex_value = line.split('(_sex')[1].split(')')[0].strip()
        # 转换_sex字段的值为对应的数字
        if sex_value == 'male':
            sex_value = 1
        elif sex_value == 'female':
            sex_value = 0
        # 添加到数据列表中
        data.append([sex_value])

# 保存数据为CSV文件
with open('DSsex.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
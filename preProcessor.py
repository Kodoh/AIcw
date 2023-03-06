import csv
import random
rows = []
minMaxValuesTraining = [[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000]]
minMaxValuesTesting = [[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000],[10000,-10000]]
training = []
validation = []
test = []
with open('DataSet.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        if not(row[1].isalpha() or row[2].isalpha() or row[3].isalpha() or row[4].isalpha() or row[5].isalpha() or row[0] == '62290' or row[0] == '73190'):     #Removing outliers
            if (float(row[1]) <  25 and float(row[1]) > 0 and float(row[2]) > 160 and float(row[2]) < 800 and float(row[3]) > 0):
                rows.append(row)
csv_file.close()
random.shuffle(rows)       #when data is randomised
for i in range(len(rows)): 
    for x in range(len(rows[i])):
        rows[i][x] = float(rows[i][x])
    if i < len(rows)*0.6:
        training.append(rows[i])
        for j in range(1,len(rows[i])):
            if rows[i][j] < minMaxValuesTraining[j-1][0]:
                minMaxValuesTraining[j-1][0] = rows[i][j]
            elif rows[i][j] > minMaxValuesTraining[j-1][1]:
                minMaxValuesTraining[j-1][1] = rows[i][j]
    elif i >= len(rows)*0.6 and i <= len(rows)*0.8:
        validation.append(rows[i])
        for j in range(1,len(rows[i])):
            if rows[i][j] < minMaxValuesTraining[j-1][0]:
                minMaxValuesTraining[j-1][0] = rows[i][j]
            elif rows[i][j] > minMaxValuesTraining[j-1][1]:
                minMaxValuesTraining[j-1][1] = rows[i][j]
    else:
        test.append(rows[i])
        for j in range(1,len(rows[i])):
            if rows[i][j] < minMaxValuesTesting[j-1][0]:
                minMaxValuesTesting[j-1][0] = rows[i][j]
            elif rows[i][j] > minMaxValuesTesting[j-1][1]:
                minMaxValuesTesting[j-1][1] = rows[i][j] 


for i in training:
    for j in range(1,7):
        i[j] = 0.8*((float(i[j]) - minMaxValuesTraining[j-1][0])/(minMaxValuesTraining[j-1][1] - minMaxValuesTraining[j-1][0])) + 0.1

for i in validation:
    for j in range(1,7):
        i[j] = 0.8*((float(i[j]) - minMaxValuesTraining[j-1][0])/(minMaxValuesTraining[j-1][1] - minMaxValuesTraining[j-1][0])) + 0.1

for i in test:
    for j in range(1,7):
        i[j] = 0.8*((float(i[j]) - minMaxValuesTesting[j-1][0])/(minMaxValuesTesting[j-1][1] - minMaxValuesTesting[j-1][0])) + 0.1



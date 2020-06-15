'''
import csv
import os
# csv_file = input('Enter the name of input file: ')
# txt_file = input('Enter the name of the output file: ')


arr = os.listdir('./CSV_text')
#arr = os.listdir('src/text-classification-on-embedding/data/political-data')
print(arr)

path = './CSV_text/'
#path = 'src/text-classification-on-embedding/data/political-data/'
for input_file in arr:
    output_file = path + input_file[:-4]
    csv_file = path + input_file
    with open(output_file, "w") as output_file:
        with open(csv_file, "r") as input_file:
            [output_file.write("".join(row)+'\n')
             for row in csv.reader(input_file)]
            os.remove(csv_file)
            output_file.close()
'''



##Text file to CSV conversion

'''
import pandas as pd
df = pd.read_fwf('test.txt')
df.to_csv('test.csv')
'''

##OR

'''
import csv

with open('test.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\n") for line in stripped if line)
    with open('test.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        #writer.writerow(('title', 'intro'))
        writer.writerows(lines)
'''
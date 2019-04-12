import os
# 017656917552
r_filepath='spd.tmp'
w_filepath='spd'
lineList=[]
with open(r_filepath) as fp:
    line = fp.readline()
    while line:
        if len(line) > 8:
            lineList.append(line) 
        line = fp.readline()

with open(w_filepath, mode='wt', encoding='utf-8') as myfile:
    myfile.write(''.join(lineList))

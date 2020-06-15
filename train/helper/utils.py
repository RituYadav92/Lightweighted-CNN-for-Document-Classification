import os

# Preprocess
# r_filepath='spd.tmp'
# w_filepath='spd'
# lineList=[]
# with open(r_filepath) as fp:
#     line = fp.readline()
#     while line:
#         if len(line) > 8:
#             lineList.append(line) 
#         line = fp.readline()

# with open(w_filepath, mode='wt', encoding='utf-8') as myfile:
#     myfile.write(''.join(lineList))
# Preprocess   

def flatten_list(l):
    '''flatteinig list of list to a list'''
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

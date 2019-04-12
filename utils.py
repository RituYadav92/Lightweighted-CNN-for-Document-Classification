def flatten_list(l):
    '''flatteinig list of list to a list'''
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

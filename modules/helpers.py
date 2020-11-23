import pandas as pd

def get_int_uid(meta, string_UIDs):
    if isinstance(meta, pd.core.series.Series):
        return meta['int_uid']
    else:
        return meta.loc[string_UIDs]['int_uid'].tolist()

def get_string_uid(meta, int_UIDs):
    # Could probably be neater
    # Accepts list of ints and single int
    
    # If meta is just a row the answer is simple
    if isinstance(meta, pd.core.series.Series):
        return meta.name
    
    # Handling lists of int_UIDs
    if not isinstance(int_UIDs, list):
        int_UIDs = [int_UIDs]
    l = []
    for int_UID in int_UIDs:
        l.append(meta[meta['int_uid'] == int_UID].index.tolist()[0])
    if len(l) == 1:
        return l[0]
    else:
        return l
     
def count_images_in_partition(meta, train_val_test):
    try: 
        n_images = meta['train_val_test'].value_counts()[train_val_test]
    except KeyError as e:
        n_images = 0
    return n_images

def count_tiles_in_partition(meta, train_val_test):
    try: 
        n_tiles = sum(meta.loc[meta['train_val_test'] == train_val_test, 'n_tiles'])
    except KeyError as e:
        n_tiles = 0
    return n_tiles
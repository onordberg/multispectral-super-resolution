def get_int_uid(meta, string_UIDs):
    return meta.loc[string_UIDs]['int_uid'].tolist()

def get_string_uid(meta, int_UIDs):
    # Could probably be neater
    # Accepts list of ints and single int
    if not isinstance(int_UIDs, list):
        int_UIDs = [int_UIDs]
    l = []
    for int_UID in int_UIDs:
        l.append(meta[meta['int_uid'] == int_UID].index.tolist()[0])
    if len(l) == 1:
        return l[0]
    else:
        return l
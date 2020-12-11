import pandas as pd
import pickle
import pathlib


def get_int_uid(meta, string_UIDs):
    if isinstance(meta, pd.core.series.Series):
        return meta['int_uid']
    else:
        return meta.loc[string_UIDs]['int_uid'].tolist()


def get_string_uid(meta, int_UIDs):
    # If meta is just a row the answer is simple
    if isinstance(meta, pd.core.series.Series):
        return meta.name

    # Handling lists of int_UIDs
    if not isinstance(int_UIDs, list):
        int_UIDs = [int_UIDs]
    string_uids = []
    for int_UID in int_UIDs:
        string_uids.append(meta[meta['int_uid'] == int_UID].index.tolist()[0])
    if len(string_uids) == 1:
        return string_uids[0]
    else:
        return string_uids


def get_sensor(meta, string_UIDs):
    if isinstance(string_UIDs, str):
        return meta.loc[string_UIDs, 'sensorVehicle']
    elif isinstance(string_UIDs, list):
        return meta.loc[string_UIDs, 'sensorVehicle'].tolist()


def count_images_in_partition(meta, train_val_test):
    try:
        n_images = meta['train_val_test'].value_counts()[train_val_test]
    except KeyError:
        n_images = 0
    return n_images


def count_images(meta):
    return len(meta.index)


def count_tiles_in_partition(meta, train_val_test):
    try:
        n_tiles = sum(meta.loc[meta['train_val_test'] == train_val_test, 'n_tiles'])
    except KeyError:
        n_tiles = 0
    return n_tiles


def count_tiles(meta):
    return sum(meta['n_tiles'])


def subset_by_areas_sensor(meta, areas=None, sensors=None):
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(areas, list):
        meta = meta.loc[meta['area_name'].isin(areas)]
    if isinstance(sensors, str):
        sensors = [sensors]
    if isinstance(sensors, list):
        meta = meta.loc[meta['sensorVehicle'].isin(sensors)]
    return meta


def load_meta_pickle_csv(dir_path, name, from_pickle=True, from_csv=False):
    if isinstance(dir_path, str):
        dir_path = pathlib.Path(dir_path)
    path = dir_path.joinpath(name)
    if from_pickle:
        if path.suffix != '.pickle':
            path = path.with_suffix('.pickle')
        with open(path, 'rb') as file:
            meta = pickle.load(file)
    if from_csv:
        raise NotImplementedError
    return meta


def save_meta_pickle_csv(meta, dir_path, name, to_pickle=True, to_csv=True):
    if isinstance(dir_path, str):
        dir_path = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path.joinpath(name)
    if to_pickle:
        if path.suffix != '.pickle':
            path = path.with_suffix('.pickle')
        with open(path, 'wb') as file:
            pickle.dump(meta, file)
        print('Saved metadata dataframe as a pickle at', path)
    if to_csv:
        if path.suffix != '.csv':
            path = path.with_suffix('.csv')
        meta.to_csv(path)
        print('Saved metadata dataframe as csv at', path)


def list_tiles_in_dir(dir_path, ms_or_pan='pan'):
    if isinstance(dir_path, str):
        dir_path = pathlib.Path(dir_path)
    list_of_tiles = list(dir_path.glob('**/*'+str(ms_or_pan)+'/*.tif'))
    print('Found', len(list_of_tiles), 'tiles of type', ms_or_pan, 'in the directory provided')
    return list(dir_path.glob('**/*'+str(ms_or_pan)+'/*.tif'))

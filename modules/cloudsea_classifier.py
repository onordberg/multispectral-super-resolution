from modules.image_utils import *
from modules.helpers import *


def load_and_populate_label_df(label_df_path, meta):
    # Reading the labels from csv
    label_df = pd.read_csv(pathlib.Path(label_df_path), delimiter=';', )

    # Extract information from the tile_uids
    tile_int_uids = [int(tile_uid.split('-')[0]) for tile_uid in label_df['tile_uid'].to_list()]
    tile_size = [int(tile_uid.split('-')[1]) for tile_uid in label_df['tile_uid'].to_list()]
    tile_string_uids = get_string_uid(meta, tile_int_uids)
    tile_sensors = get_sensor(meta, tile_string_uids)

    # Add the extracted information as columns
    label_df['sensor'] = tile_sensors
    label_df['tile_size'] = tile_size
    label_df['img_uid'] = tile_string_uids
    return label_df


def cloudsea_preprocess(pan_img):
    MODEL_INPUT_SIZE = 224  # efficientnet-b0 design size
    pan_img = tf.image.resize(pan_img, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], method='bicubic').numpy()
    return pan_img


def prepare_for_training(label_df, tif_paths_pan, tif_paths_ms,
                         pan_or_ms_or_both='pan',
                         pan_tile_size=128, ms_tile_size=32, resize_method='bilinear'):
    # Assert that uids match up (that the order has not been mixed up etc.)
    assert_x_y_match(tif_paths_pan=tif_paths_pan, tif_paths_ms=tif_paths_ms, label_df=label_df)

    ms_channels = 4  # The number of bands in the GE01 sensor (least of GE01, WV02 and WV03_VNIR)

    # y: Prediction classes 0, 1
    n = len(label_df)
    y = label_df['cloud-sea'].to_numpy(dtype=np.uint16)

    if pan_or_ms_or_both == 'pan':
        tile_size = pan_tile_size
        channels = 1
    elif pan_or_ms_or_both == 'ms':
        tile_size = ms_tile_size
        channels = ms_channels
    elif pan_or_ms_or_both == 'both':
        tile_size = pan_tile_size
        channels = ms_channels + 1  # + 1 is the pan channel
    else:
        raise ValueError('Argument pan_or_ms_or_both must be either "pan", "ms" or "both", not', pan_or_ms_or_both)

    X = np.empty((n, tile_size, tile_size, channels), dtype=np.uint16)
    sensors = label_df['sensor'].to_list()

    for i in range(n):
        sensor = sensors[i]
        band_indices = None
        if pan_or_ms_or_both in ['ms', 'both']:
            if sensor in ['WV02', 'WV03_VNIR']:
                band_indices = sensor_a_imitate_sensor_b(sensor_a_name=sensor, sensor_b_name='GE01')
                assert len(band_indices) == ms_channels
            elif sensor == 'GE01':
                band_indices = [0, 1, 2, 3]
            else:
                raise NotImplementedError('Sensor', sensor, 'not implemented. Only WV02, GE01 and WV03_VNIR.')

        if pan_or_ms_or_both == 'pan':
            img = geotiff_to_ndarray(tif_paths_pan[i])
            img = tf.image.resize(img, [tile_size, tile_size],
                                  method=resize_method).numpy()
        elif pan_or_ms_or_both == 'ms':
            img = geotiff_to_ndarray(tif_paths_ms[i])
            img = select_bands(img, band_indices)
            img = tf.image.resize(img, [tile_size, tile_size],
                                  method=resize_method).numpy()
        elif pan_or_ms_or_both == 'both':
            img_ms = geotiff_to_ndarray(tif_paths_ms[i])
            img_ms = select_bands(img_ms, band_indices)
            img_ms = tf.image.resize(img_ms, [tile_size, tile_size],
                                     method=resize_method).numpy()
            img_pan = geotiff_to_ndarray(tif_paths_pan[i])
            img_pan = tf.image.resize(img_pan, [tile_size, tile_size],
                                      method=resize_method).numpy()
            img = np.concatenate((img_pan, img_ms), axis=-1)
        else:
            raise ValueError('Argument pan_or_ms_or_both must be either "pan", "ms" or "both", not', pan_or_ms_or_both)

        X[i,:,:,:] = img
        if i % 100 == 0:
            print('Loaded', i, 'images')
    print('Finished preparing', n, 'images and labels for training!')
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    return X, y


def assert_x_y_match(tif_paths_pan, tif_paths_ms, label_df):
    label_tile_uid, tif_tile_ms_uid, tif_tile_pan_uid = None, None, None
    try:
        assert len(label_df) == len(tif_paths_pan) == len(tif_paths_ms)
    except AssertionError:
        print('Lengths of tif paths and label dataframe differ!')

    n = len(label_df)
    try:
        for i in range(n):
            label_tile_uid = label_df.iloc[i]['tile_uid']
            tif_tile_ms_uid = tif_paths_ms[i].stem
            tif_tile_pan_uid = tif_paths_pan[i].stem
            # print(label_tile_uid, tif_tile_ms_uid, tif_tile_pan_uid)
            assert label_tile_uid == tif_tile_ms_uid == tif_tile_pan_uid
    except AssertionError:
        print('Mismatch between sequence of tile uids!')
        print('label_tile_uid:', label_tile_uid)
        print('tif_tile_ms_uid', tif_tile_ms_uid)
        print('tif_tile_pan_uid', tif_tile_pan_uid)
    print('Verification OK. All', n, 'image tile UIDs match.')


def build_augmentation_model():
    img_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=1.0),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.experimental.preprocessing.RandomFlip(),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )
    return img_augmentation


def build_model(augment=True, input_shape=(224, 224, 1), learning_rate=0.001):
    inputs = tf.keras.layers.Input(shape=input_shape)

    if augment:
        x = build_augmentation_model()(inputs)
    else:
        x = inputs

    model = tf.keras.applications.EfficientNetB0(
        include_top=True, weights=None, input_tensor=x,
        pooling=None, classes=1, classifier_activation='sigmoid')

    model = tf.keras.Model(inputs, model.output, name="EfficientNetB0")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss='binary_crossentropy', metrics='accuracy')

    return model

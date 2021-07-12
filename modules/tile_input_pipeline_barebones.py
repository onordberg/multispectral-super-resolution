class GeotiffDataset:
    def __init__(self, ms_tile_shape=(32, 32, 8), pan_tile_shape=(128, 128, 1),
                 sensor='WV02', band_selection=(1, 2, 4, 6),
                 augment_flip=False, augment_rotate=False):
        self.ms_tile_shape = ms_tile_shape
        self.pan_tile_shape = pan_tile_shape
        self.sensor = sensor
        self.band_selection = band_selection
        self.augment_flip = augment_flip
        self.augment_rotate = augment_rotate
        self.dataset = self.build_dataset()

    def get_dataset(self):
        return self.dataset

    def build_dataset(self):
        # Find and list all MS patches in the specified directory
        file_pattern = str(pathlib.Path(self.tiles_path).joinpath(
            self.sensor + '*', 'ms*', '*.tif'))
        ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)

        # Turn list of files into paired arrays of MS and PAN patches
        ds = ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        ds = ds.repeat()

        if self.augment_flip:
            ds = ds.map(self.random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.random_rotate:
            ds = ds.map(self.random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def process_path(self, ms_tile_path):
        ms_img = tf.py_function(self.decode_geotiff, [ms_tile_path, 'ms'], [tf.float32])
        pan_tile_path = tf.strings.regex_replace(ms_tile_path, '\\\\ms\\\\', '\\\\pan\\\\')
        pan_img = tf.py_function(self.decode_geotiff, [pan_tile_path, 'pan'], [tf.float32])
        return ms_img, pan_img

    def decode_geotiff(self, image_path, ms_or_pan):
        with rasterio.open(image_path) as src:
            img = src.read()
        # Select bands from the ms patch
        if ms_or_pan == 'ms':
            img = self.select_bands(img, self.band_selection)
        img = input_scaler(img, radius=1.0, output_dtype='float32',
                           uint_bit_depth=11, mean=self.mean_correction)
        return img

    def select_bands(imgs, band_indices):
        return np.take(imgs, band_indices, axis=-1)

    def random_flip(self, ms, pan):
        flip_left_right = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        flip_up_down = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        if flip_left_right == 1:
            ms_img = tf.image.flip_left_right(ms)
            pan_img = tf.image.flip_left_right(pan)
        if flip_up_down == 1:
            ms_img = tf.image.flip_up_down(ms)
            pan_img = tf.image.flip_up_down(pan)
        return ms, pan

    def random_rotate(self, ms, pan):
        rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        ms_img = tf.image.rot90(ms, k=rotation)
        pan_img = tf.image.rot90(pan, k=rotation)
        return ms, pan

ds_train = GeotiffDataset(tiles_path=TILES_PATH, batch_size=16,
                          ms_tile_shape=(32, 32, 4), pan_tile_shape=(128, 128, 1),
                          sensor='WV02', band_selection=(1, 2, 4, 6),
                          mean_correction=MEAN, cache_memory=True,
                          cache_file=CACHE_PATH, repeat=True,
                          shuffle=True, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE)
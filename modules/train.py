from modules.logging import *


def pretrain_esrgan(generator,
                    ds_train_dict,
                    epochs,
                    steps_per_epoch,
                    initial_epoch=0,
                    validate=False,
                    ds_val_dict=None,
                    val_steps=250,
                    model_name=None,
                    tag=None,
                    log_tensorboard=False,
                    tensorboard_logs_dir='logs/tb/',
                    save_models=False,
                    models_save_dir='logs/models',
                    save_weights_only=True,
                    log_train_images=False,
                    n_train_image_batches=1,
                    log_val_images=False,
                    n_val_image_batches=1):

    logger = EsrganLogger(
        model_name=model_name,
        tag=tag,
        log_tensorboard=log_tensorboard,
        tensorboard_logs_dir=tensorboard_logs_dir,
        save_models=save_models,
        models_save_dir=models_save_dir,
        save_weights_only=save_weights_only,
        validate=validate,
        val_steps=val_steps,
        ds_val_dict=ds_val_dict,  # dict of dataset(s)
        log_train_images=log_train_images,
        n_train_image_batches=n_train_image_batches,
        ds_train_dict=ds_train_dict,  # dict of dataset(s)
        log_val_images=log_val_images,
        n_val_image_batches=n_val_image_batches)

    callbacks = logger.get_callbacks()
    print('Callbacks:', callbacks)

    ds_train = list(ds_train_dict.values())[0]
    history = generator.fit(ds_train,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            initial_epoch=initial_epoch,
                            callbacks=callbacks)
    return history


def gan_train_esrgan(esrgan_model,
                     ds_train_dict,
                     epochs,
                     steps_per_epoch,
                     initial_epoch=0,
                     validate=False,
                     ds_val_dict=None,
                     val_steps=250,
                     model_name=None,
                     tag=None,
                     log_tensorboard=False,
                     tensorboard_logs_dir='logs/tb/',
                     save_models=False,
                     models_save_dir='logs/models',
                     save_weights_only=True,
                     log_train_images=False,
                     n_train_image_batches=1,
                     log_val_images=False,
                     n_val_image_batches=1):

    logger = EsrganLogger(
        model_name=model_name,
        tag=tag,
        log_tensorboard=log_tensorboard,
        tensorboard_logs_dir=tensorboard_logs_dir,
        save_models=save_models,
        models_save_dir=models_save_dir,
        save_weights_only=save_weights_only,
        validate=validate,
        val_steps=val_steps,
        ds_val_dict=ds_val_dict,  # dict of dataset(s)
        log_train_images=log_train_images,
        n_train_image_batches=n_train_image_batches,
        ds_train_dict=ds_train_dict,  # dict of dataset(s)
        log_val_images=log_val_images,
        n_val_image_batches=n_val_image_batches)

    callbacks = logger.get_callbacks()
    print('Callbacks:', callbacks)

    ds_train = list(ds_train_dict.values())[0]
    history = esrgan_model.fit(ds_train,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               initial_epoch=initial_epoch,
                               callbacks=callbacks)
    return history

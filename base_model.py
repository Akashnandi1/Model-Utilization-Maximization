import tensorflow as tf

def select_model(cfg, total_classes):
    if cfg['model']['architecture'] == 'eb0':
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                       weights='imagenet',
                                                                       input_tensor=None,
                                                                       input_shape=(cfg['model']['img_size'], cfg['model']['img_size'], 3),
                                                                       pooling='avg')

        x = tf.keras.layers.Dropout(0.2)(base_model.layers[-1].output)
        x = tf.keras.layers.Dense(total_classes,
                                  activation = 'softmax',
                                  kernel_initializer = 'glorot_uniform',
                                  bias_initializer = 'Zeros')(x)
        model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
        return model

    elif cfg['model']['architecture'] == 'eb1':
        base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False,
                                                                       weights='imagenet',
                                                                       input_tensor=None,
                                                                       input_shape=(cfg['model']['img_size'], cfg['model']['img_size'], 3),
                                                                       pooling='avg')

        x = tf.keras.layers.Dropout(0.2)(base_model.layers[-1].output)
        x = tf.keras.layers.Dense(total_classes,
                                  activation = 'softmax',
                                  kernel_initializer = 'glorot_uniform',
                                  bias_initializer = 'Zeros')(x)
        model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
        return model

    elif cfg['model']['architecture'] == 'eb2':
        base_model = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False,
                                                                       weights='imagenet',
                                                                       input_tensor=None,
                                                                       input_shape=(cfg['model']['img_size'], cfg['model']['img_size'], 3),
                                                                       pooling='avg')

        x = tf.keras.layers.Dropout(0.2)(base_model.layers[-1].output)
        x = tf.keras.layers.Dense(total_classes,
                                  activation = 'softmax',
                                  kernel_initializer = 'glorot_uniform',
                                  bias_initializer = 'Zeros')(x)
        model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
        return model

    elif cfg['model']['architecture'] == 'eb3':
        base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,
                                                                       weights='imagenet',
                                                                       input_tensor=None,
                                                                       input_shape=(cfg['model']['img_size'], cfg['model']['img_size'], 3),
                                                                       pooling='avg')

        x = tf.keras.layers.Dropout(0.2)(base_model.layers[-1].output)
        x = tf.keras.layers.Dense(total_classes,
                                  activation = 'softmax',
                                  kernel_initializer = 'glorot_uniform',
                                  bias_initializer = 'Zeros')(x)
        model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
        return model
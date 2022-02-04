

import tensorflow as tf

"""# Model

## Convolutional layers
"""

def get_down_conv2d(filter, size, batchNorm = True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2D(
        filter, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False)
    )

    if batchNorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def get_up_conv2d_transpose(filters, size, applyDropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2,
        padding='same',
        kernel_initializer=initializer,
        use_bias=False)
    )

    result.add(tf.keras.layers.BatchNormalization())

    if applyDropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

"""## Generator"""

def Generator():
    input1 = tf.keras.layers.Input(shape=[256,256,3])
    input2 = tf.keras.layers.Input(shape=[256,256,3])

    initializer = tf.random_normal_initializer(0., 0.02)

    encoder = [
        get_down_conv2d(3, 4, batchNorm=False),
        get_down_conv2d(64, 4),
        get_down_conv2d(64, 4), 
        get_down_conv2d(128, 4), 
        get_down_conv2d(256, 4), 
        get_down_conv2d(256, 4), 
        get_down_conv2d(512, 4), 
        get_down_conv2d(512, 4), 
        get_down_conv2d(1024, 4), 
    ]

    decoder = [
        get_up_conv2d_transpose(512,4, applyDropout=True),
        get_up_conv2d_transpose(512,4, applyDropout=True),
        get_up_conv2d_transpose(256,4, applyDropout=True),
        get_up_conv2d_transpose(256,4),
        get_up_conv2d_transpose(128,4),
        get_up_conv2d_transpose(64,4),
        get_up_conv2d_transpose(64,4),    
            
    ]

    last = tf.keras.layers.Conv2DTranspose(
        3, # RGB
        4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )

    x = tf.keras.layers.concatenate([input1, input2])
    skip_connection = []

    for layer in encoder:
        x = layer(x)
        skip_connection.append(x)

    skip_connection = reversed(skip_connection[:-2])

    for layer, skip in zip(decoder, skip_connection): #U-Net
        x = layer(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[input1,input2], outputs=x)


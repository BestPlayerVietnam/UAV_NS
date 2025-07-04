from keras import layers, models
from keras.applications import MobileNetV2
from config import Config

def ASPP(x, out_channels):
    # Atrous Spatial Pyramid Pooling
    shape = x.shape

    y1 = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation("relu")(y1)

    y2 = layers.Conv2D(out_channels, 3, dilation_rate=3, padding="same", use_bias=False)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.Activation("relu")(y2)

    y3 = layers.Conv2D(out_channels, 3, dilation_rate=5, padding="same", use_bias=False)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.Activation("relu")(y3)

    y4 = layers.Conv2D(out_channels, 3, dilation_rate=7, padding="same", use_bias=False)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.Activation("relu")(y4)

    y5 = layers.GlobalAveragePooling2D()(x)
    y5 = layers.Reshape((1, 1, shape[-1]))(y5)
    y5 = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y5)
    y5 = layers.BatchNormalization()(y5)
    y5 = layers.Activation("relu")(y5)
    y5 = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y5)

    y = layers.Concatenate()([y1, y2, y3, y4, y5])
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    return y

def deeplab(input_shape=(288, 288, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    # Извлекаем фичи
    high_level_feature = base_model.get_layer("block_13_expand_relu").output  # stride=16
    low_level_feature = base_model.get_layer("block_3_expand_relu").output    # stride=4

    # ASPP
    x = ASPP(high_level_feature, 256)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Обработка low-level features
    low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level_feature)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.Activation("relu")(low_level)
    low_level = layers.Dropout(0.1)(low_level)

    # Конкатенация
    x = layers.Concatenate()([x, low_level])

    # Финальные сверточные блоки
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    # Восстановление размера до исходного
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Выходной слой
    outputs = layers.Conv2D(Config.NUM_CLASSES, kernel_size=1, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

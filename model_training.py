import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from config import base_dir, img_size, batch_size

def train_model():
    """Trains an EfficientNet-based CNN model for Aadhaar classification."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=30, zoom_range=0.2,
        horizontal_flip=True, validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation'
    )

    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')
    model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[checkpoint])

    model.save("final_model.keras")

if __name__ == "__main__":
    train_model()

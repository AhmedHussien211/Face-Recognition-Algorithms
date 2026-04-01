from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from deepface import create_deepface

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    "../../Datasets/train",
    target_size=(152,152),
    batch_size=64,
    class_mode='categorical',
    subset='training')

val_gen = train_datagen.flow_from_directory(
    "../../Datasets/train",
    target_size=(152,152),
    batch_size=64,
    class_mode='categorical',
    subset='validation')

num_classes = len(os.listdir("../../Datasets/train"))  # number of persons
model = create_deepface((152,152,3), num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_gen, validation_data=val_gen, epochs=15)

model.save("deepface.h5")


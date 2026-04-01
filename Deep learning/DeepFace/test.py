from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = load_model("deepface.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    "../../Datasets/test",
    target_size=(152,152),
    batch_size=64,
    class_mode='categorical',
    shuffle=False)

loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)


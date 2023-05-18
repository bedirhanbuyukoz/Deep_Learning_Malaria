import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Veri yolları
train_data_dir = 'training_set'  # Replace with the actual directory path to your train data
validation_data_dir = 'single_prediction'  # Replace with the actual directory path to your validation data
test_data_dir = 'testing_set'  # Replace with the actual directory path to your test data

# Hyperparameters
num_classes = 2
batch_size = 32
epochs = 3  # Half of the original epoch value

# Veri ön işleme ve genişletme
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Veri yükleyiciler
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Önceden eğitilmiş MobileNet modelini yükleyin
base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Yeni katmanları ekleyin
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Yeni modeli oluşturun
model = Model(inputs=base_model.input, outputs=predictions)

# Yeni katmanların eğitilmesini sağlayın
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Test verileri üzerinde modeli değerlendirin
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Tahminleri alın
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Gerçek etiketleri alın
y_true = test_generator.classes

# Sınıflandırma raporu ve karmaşıklık matrisini hesaplayın
class_labels = list(test_generator.class_indices.keys())
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_labels))

print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

# Başarıyı artırmak için modeli ince ayar yapın
# İlk olarak, tüm katmanları eğitilebilir hale getirin
for layer in model.layers:
    layer.trainable = True

# İkinci olarak, düzenleyici yöntemler uygulayın
# Örnek olarak, dropout kullanalım
from tensorflow.keras.layers import Dropout

# Dropout katmanını ekleyin
model.add(Dropout(0.5))

# Modeli tekrar derleyin
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli daha uzun süre eğitin
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Test verileri üzerinde modeli değerlendirin
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Veri yolları
train_data_dir = 'training_set'  # Eğitim veri setinin gerçek dizin yolunu belirtme
validation_data_dir = 'single_prediction'  # Doğrulama veri setinin gerçek dizin yolunu belirtme
test_data_dir = 'testing_set'  # Test veri setinin gerçek dizin yolunu belirtme

# Hyperparameters
num_classes = 2
batch_size = 128
epochs = 5

# Veri ön işleme ve genişletme
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

# Önceden eğitilmiş MobileNet modelini yükleme
base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Yeni katmanları ekleme
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Yeni modeli oluşturma
model = Model(inputs=base_model.input, outputs=predictions)

# Yeni katmanların eğitilmesini sağlama
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Test verileri üzerinde modeli değerlendirme
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',   shuffle=False
)

# Tahminleri alma
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Gerçek etiketleri alma
y_true = test_generator.classes

# Sınıflandırma raporu ve karmaşıklık matrisini hesaplama
class_labels = list(test_generator.class_indices.keys())
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_labels))

print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))

# Başarıyı artırmak için modeli ince ayar yapma
# İlk olarak, tüm katmanları eğitilebilir hale getirme
for layer in model.layers:
    layer.trainable = True

# Modeli tekrar derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli daha uzun süre eğitme
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Test verileri üzerinde modeli değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Eğitim sürecinin kaybını çizme
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history_finetune.history['loss'])  # İnce ayar sonrası kayıpları da çizme
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Fine-tuning'], loc='upper left')

# Eğitim sürecinin doğruluğunu çizme
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history_finetune.history['accuracy'])  # İnce ayar sonrası doğrulukları da çizme
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Fine-tuning'], loc='upper left')
plt.tight_layout()
plt.show()

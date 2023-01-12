import os
import cv2
import numpy as np 
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt



data_dir = 'GSRTB'
train_path = 'GTSRB/Train'
test_path = 'GTSRB/Test/'

folders = os.listdir(train_path)
samples_dict = {} 

print("Veri Seti Hazırlanıyor...")
for folder in folders:
    images_in_folder = os.listdir(train_path + '/' + folder)
    samples_dict[folder] = len(images_in_folder)

image_data = []
image_labels = []


class_num = len(os.listdir(train_path))
for i in range(class_num):
    path = train_path +'/'+ str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB') 
            resize_image = image_fromarray.resize((32, 32)) #Resimler 32x32 boyutuna getiriliyor
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)


image_data = np.array(image_data)
image_labels = np.array(image_labels)
print(image_data.shape, image_labels.shape)

print("Veri Seti Eğitim ve Test Verilerine Bölünüyor...")
x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.2, random_state=42, shuffle=True)

print("Veri Seti Normalizasyon İşlemi Gerçekleştiriliyor...")
x_train = x_train/255.0
x_test = x_test/255.0

print("Veri Seti One-Hot Encoding İşlemi Gerçekleştiriliyor...")
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


print("Eğitim ve test veriseti boyutları: ")
print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)


resnet_model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=43)
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = resnet_model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = resnet_model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = np.sum(predictions==y_true)/len(y_true)
print("Accuracy:", accuracy)

print("Model Kaydediliyor...")
resnet_model.save('traffic_sign_classifier_resnet.h5')

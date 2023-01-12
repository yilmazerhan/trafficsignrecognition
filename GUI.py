import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import tensorflow as tf

# Daha önce eğitilen modeli yükle
model = tf.keras.models.load_model('traffic_sign_classifier.h5')

# Etiketler
classes = { 
    1:'Hız Sınırı (20km/h)',
    2:'Hız Sınırı (30km/h)',
    3:'Hız Sınırı (50km/h)',
    4:'Hız Sınırı (60km/h)',
    5:'Hız Sınırı (70km/h)',
    6:'Hız Sınırı (80km/h)',
    7:'Hız Sınırı Bitişi (80km/h)',
    8:'Hız Sınırı (100km/h)',
    9:'Hız Sınırı (120km/h)',
    10:'Geçilmez',
    11:'3.5 ton ve üzeri araçlar geçemez',
    12:'Kavşakta geçiş hakkı',
    13:'Öncelikli yol',
    14:'Yol Ver',
    15:'Dur',
    16:'Araçlar Giremez',
    17:'3.5 ton ve üzeri araçlar giremez',
    18:'Girilmez',
    19:'Dikkat!',
    20:'Sola Tehlikeli Viraj',
    21:'Sağa Tehlikeli Viraj',
    22:'Viraj',
    23:'Engebeli Yol',
    24:'Kaygan Yol',
    25:'Yol sağda daralıyor',
    26:'Yol Çalışması',
    27:'Trafik Lambası',
    28:'Yaya Yolu',
    29:'Çocuk Geçişi',
    30:'Bisiklet Geçişi',
    31:'Buz ve Karlı Yol',
    32:'Vahşi Hayvan Geçebilir',
    33:'Hız ve Giçiş Limiti Sonu',
    34:'Sağa Dönüş',
    35:'Sola Dönüş',
    36:'Düz İlerleyin',
    37:'İleri veya Sağa',
    38:'İleri veya Sola',
    39:'Sağdan Devam Edin',
    40:'Soldan Devam Edin',
    41:'Ada Etrafında Dön',
    42:'Geçiş Limiti Sonu',
    43:'3.5 ton üzeri araçlar için geçiş limiti sonu' 
    }

# Resim dosyasının yüklenmesi
def select_image():
    file_path = filedialog.askopenfilename()
    secilen_resim['text'] = file_path

# Yüklenen resim dosyasının modelin kullanılarak sınıflandırılması
def classify_image():
    image = tf.keras.preprocessing.image.load_img(secilen_resim['text'], target_size=(30, 30)) # Seçilen resmin yüklenmesi ve modele uygun boyuta getirilmesi
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    # model kullanılarak resmin sınıflandırılması
    prediction = model.predict(image_array)
    prediction_label = np.argmax(prediction)
    sonuc['text'] = f'Tahmin edilen: {classes[prediction_label+1]}'

# GUI penceresini aç
ana_ekran = tk.Tk()
ana_ekran.geometry('600x200')
ana_ekran.title('Eğitilen Model ile Trafik İşaretini Bul')
ana_ekran.configure(background='#e9ecef')

# Dosya yükleme butonu
dosya_yukle = tk.Label(ana_ekran, text='Bir resim dosyası seçin', font=('arial',12,'bold'))
dosya_yukle_butonu = tk.Button(ana_ekran, text='Yükle', font=('arial',12,'bold'), command=lambda: select_image(), bg='#4285F4', fg='white')

# Yüklenen Resim dosyasını ekrana yazdır
secilen_resim = tk.Label(ana_ekran, text='Resim yüklenmedi')

# Tahmin et butonu
tahmin_butonu = tk.Button(ana_ekran, text='Trafik İşaretini Tahmin Et', font=('arial',12,'bold'), command=lambda: classify_image(), bg='#DB4437', fg='white')

# Sonucu ekrana yazdır
sonuc = tk.Label(ana_ekran, text='', font=('arial',15,'bold'))


# Ekranı oluştur
dosya_yukle.pack()
dosya_yukle_butonu.pack()
secilen_resim.pack()
tahmin_butonu.pack()
sonuc.pack()

ana_ekran.mainloop()


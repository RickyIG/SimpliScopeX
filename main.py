import streamlit as st
import lorem
from numerize import numerize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
# import streamlit_book as stb
import plotly.express as px
import plotly

from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time
import cv2


st.set_page_config(layout="wide")


st.title("*SimpliScopeX*: Model *Deep Learning* yang Ditingkatkan untuk Identifikasi Citra Mikroskopis Fragmen Simplisia Daun Tanaman Obat")
st.write("")
st.write("")

col1, mid, col2 = st.columns([1,1,35])
with col1:
    st.image('./assets/img/circle.png', width=60)
with col2:
    st.markdown("**Ricky Indra Gunawan**<br>197006029", unsafe_allow_html=True)
st.markdown("---")

st.write("""Seiring dengan berkembangnya tren “Back to Nature”, masyarakat mulai beralih ke pengobatan herbal atau pengobatan tradisional yang berasal dari alam. Kebenaran serbuk simplisia kering tanaman obat salah satunya dapat ditentukan melalui uji mikroskopis dengan melihat fragmen-fragmen pengenalnya. Maka dari itu, peneliti mengusulkan penelitian sebagai langkah inovatif melalui penerapan teknologi untuk melakukan identifikasi fragmen mikroskopis simplisia daun kering dan nama dari spesies tanaman obat.""")

st.subheader("Identifikasi Citra Mikroskopis Fragmen Simplisia Daun Tanaman Obat")
st.write("Masukkan gambar (minimal ukuran gambar: 480x480 pixel)")

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def predicting(image):
    classifier_model = "./model/final_simpliscopex_model.h5"
      
    model = load_model(classifier_model)

    # img = tf.keras.utils.load_img(image)
    plt.figure()
    plt.rcParams["figure.autolayout"] = True
    f, axarr = plt.subplots(1,2, figsize=(12, 8))
    plt.subplots_adjust(left=0.1,
                        right=0.2,
                        wspace=0.4)
    axarr[0].imshow(image)
    axarr[0].set_title('Original Image')

    # # image = tf.image.decode_jpeg(image, channels=3)
    # image_input = image_input.convert('RGB')
    
    # # Convert the PIL image to a byte stream encoded as JPEG
    # byte_io = io.BytesIO()
    # image_input.save(byte_io, format='JPEG')
    # jpeg_bytes = byte_io.getvalue()
    
    # # Decode the JPEG byte stream using TensorFlow
    # image_tensor = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    # st.write(image)
    # st.write(type(image))
    st.write("Image shape: ", str(image.shape))
    st.write("Image dtype: ", str(image.dtype))
    img = tf.image.rgb_to_grayscale(image)
    # st.write(img.shape)
    img = center_crop(img, (480,480))
    img2 = tf.image.resize(img, (224, 224))

    # st.image(img2, caption='Processed Image')
    axarr[1].imshow(img2, cmap='gray')
    axarr[1].set_title('Processed Image')
    f.patch.set_visible(False)
    st.pyplot(f)
    
    x = tf.keras.utils.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
      
    # test_image = image.resize((224,224))
    # test_image = preprocessing.image.img_to_array(test_image)
    # test_image = test_image / 255.0
    # test_image = np.expand_dims(test_image, axis=0)
    images = np.vstack([x])
    
    # classes = load_model.predict(images, batch_size=32) 
    class_names = {0: 'katuk-bp',
                    1: 'katuk-ea_dg_palisade',
                    2: 'katuk-ea_dg_stomata',
                    3: 'katuk-eb',
                    4: 'katuk-parenkim_d_kko_b_roset',
                    5: 'keji_beling-bp',
                    6: 'keji_beling-ea',
                    7: 'keji_beling-ea_dg_litosit_d_stomata',
                    8: 'keji_beling-rp',
                    9: 'keji_beling-sistolit',
                    10: 'kelor-bp_t_tangga',
                    11: 'kelor-eb_dg_stomata',
                    12: 'kelor-kko_b_roset',
                    13: 'kelor-m_bp_dg_pt_tangga_d_kko_b_roset',
                    14: 'kelor-m_dg_selsekresi',
                    15: 'pegagan-bp',
                    16: 'pegagan-ea',
                    17: 'pegagan-eb_dg_stomata',
                    18: 'pegagan-mesofil',
                    19: 'pegagan-uratdaun_dg_kko_b_roset',
                    20: 'salam-ea',
                    21: 'salam-eb_dg_stomata',
                    22: 'salam-kko_b_prisma',
                    23: 'salam-sklerenkim',
                    24: 'salam-unsurxilem_dg_noktah',
                    25: 'sereh-e_dg_parenkim',
                    26: 'sereh-ea_d_bp_dg_p_t_tangga',
                    27: 'sereh-ea_dg_selpalisade_d_rp',
                    28: 'sereh-ea_dg_stomata_b_halter',
                    29: 'sereh-sklerenkim'}
    labels = ['katuk-bp',
                    'katuk-ea_dg_palisade',
                    'katuk-ea_dg_stomata',
                    'katuk-eb',
                    'katuk-parenkim_d_kko_b_roset',
                    'keji_beling-bp',
                    'keji_beling-ea',
                    'keji_beling-ea_dg_litosit_d_stomata',
                    'keji_beling-rp',
                    'keji_beling-sistolit',
                    'kelor-bp_t_tangga',
                    'kelor-eb_dg_stomata',
                    'kelor-kko_b_roset',
                    'kelor-m_bp_dg_pt_tangga_d_kko_b_roset',
                    'kelor-m_dg_selsekresi',
                    'pegagan-bp',
                    'pegagan-ea',
                    'pegagan-eb_dg_stomata',
                    'pegagan-mesofil',
                    'pegagan-uratdaun_dg_kko_b_roset',
                    'salam-ea',
                    'salam-eb_dg_stomata',
                    'salam-kko_b_prisma',
                    'salam-sklerenkim',
                    'salam-unsurxilem_dg_noktah',
                    'sereh-e_dg_parenkim',
                    'sereh-ea_d_bp_dg_p_t_tangga',
                    'sereh-ea_dg_selpalisade_d_rp',
                    'sereh-ea_dg_stomata_b_halter',
                    'sereh-sklerenkim']
    predictions = model.predict(images, batch_size=32)
    # scores = tf.nn.softmax(predictions[0])
    # scores = scores.numpy()
    # result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    # st.write(str(predictions)) 
    result = f"{labels[np.argmax(predictions)]}, {np.max(predictions)}"
    return result


file_uploaded = st.file_uploader("Pilih File", type=["png","jpg","jpeg"])
class_btn = st.button("Klasifikasikan!")
if file_uploaded is not None:    
    image = Image.open(file_uploaded)
    st.write(image)
    st.image(image, caption='Image yang diupload')
        
if class_btn:
    if file_uploaded is None:
        st.write("Perintah tidak valid, harap unggah gambar!")
    else:
        with st.spinner('Model sedang bekerja....'):
            plt.imshow(image)
            plt.axis("off")
            predictions = predicting(image)
            time.sleep(1)
            st.success('Telah diklasifikasi!')
            st.write(predictions)

# Plot!
st.subheader("Apakah Kamu Tahu?")
st.write("Daun manakah yang bersifat antioksidan? (Pilih 3)")
q0, q1, q2, q3, q4, q5 = st.columns(6)
with q0:
    st.image('./assets/img/katuk.jpg', width=100)
    a0 = st.checkbox("Daun Katuk")
with q1:
    st.image('./assets/img/kejibeling.jpeg', width=100)
    a1 = st.checkbox("Daun Keji Beling")
with q2:
    st.image('./assets/img/kelor.jpeg', width=100)
    a2 = st.checkbox("Daun Kelor")
with q3:
    st.image('./assets/img/pegagan.jpg', width=100)
    a3 = st.checkbox("Daun Pegagan")
with q4:
    st.image('./assets/img/salam.jpeg', width=100)
    a4 = st.checkbox("Daun Salam")
with q5:
    st.image('./assets/img/serehwangi.png', width=100)
    a5 = st.checkbox("Daun Sereh") 
if a0 and (a1 == True) and (a2 == False) and (a3 == True) and (a4 == False) and (a5 == False):
    # st.write("Anda sudah setuju")
    with st.expander("Selamat, Kamu Benar!"):
        st.write("""
            Fun Fact!
            \nDaun yang memiliki sifat antioksidan adalah daun katuk, keji beling, dan pegagan. 
            \nDaun yang memiliki sifat antibakteri adalah daun kelor, salam, dan sereh.
        """)
else:
    st.write("Anda belum menjawab/jawaban anda kurang dari 3/jawaban anda salah.")


st.write("")
st.subheader("Informasi mengenai hasil proses *training* pada model SimpliScopeX")

grafikloss = pd.read_csv("./data/log_training.csv")
grafikloss['epoch'] = grafikloss['epoch'].astype(str)
fig_ke_1 = px.line(grafikloss, x='epoch', y=['loss', 'val_loss'], title='Grafik Training Loss dan Validation Loss', markers=True)

grafikakurasi = pd.read_csv("./data/log_training.csv")
grafikakurasi['epoch'] = grafikakurasi['epoch'].astype(str)
fig_ke_2 = px.line(grafikakurasi, x='epoch', y=['accuracy', 'val_accuracy'], title='Grafik Training Accuracy dan Validation Accuracy', markers=True)


graphchart0, graphchart1 = st.columns(2)
with graphchart0:
    st.plotly_chart(fig_ke_1, use_container_width=True)
with graphchart1:
    st.plotly_chart(fig_ke_2, use_container_width=True)

metrik0, metrik1, metrik2, metrik3, metrik4 = st.columns(5)
with metrik0:
    st.metric("Training Loss", 1.1832)
with metrik1:
    st.metric("Validation Loss", 1.1912)
with metrik2:
    st.metric("Training Accuracy", "93,97%")
with metrik3:
    st.metric("Validation Accuracy", "74,49%")
with metrik4:
    st.metric("Validation Accuracy", "80,25%")

css='''
[data-testid="metric-container"] {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
}
'''

st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)

st.write("""Secara keseluruhan, model dengan akurasi terbaik didapatkan pada skenario 5 dengan training accuracy sebesar 93,97%, validation accuracy sebesar 74,49%, serta accuracy terhadap data uji sebesar 80,25%.""")
with st.columns(3)[1]:
    st.image("./assets/img/classification_report.png")
st.write("""Dapat disimpulkan bahwa bagian dari fragmen simplisia tanaman obat yang memiliki F1-score terkecil diraih oleh daun Katuk pada bagian parenkim dan kristal kalsium oksalat bentuk roset (katuk-parenkim_d_kko_b_roset) sebesar 46%, sedangkan bagian dari fragmen simplisia tanaman obat yang memiliki F1-score terbesar diraih oleh daun Sereh pada bagian epidermis atas dengan sel palisade dan rambut penutup (sereh-ea_dg_palisade_d_rp) sebesar 95%.""")
with st.columns(3)[1]:
    st.image("./assets/img/confusion_matrix.png")
st.write("""Dapat dilihat bahwa model telah melakukan prediksi terhadap data uji dengan baik dan memiliki akurasi sebesar 80,25%. Jika dilihat sekilas, terdapat sedikit penyimpangan prediksi yang dialami oleh daun kelor pada bagian mesofil, berkas pengangkut dengan penebalan tipe tangga dan kristal kalsium oksalat bentuk roset (kelor-m_bp_dg_pt_tangga_d_kko_b_roset) dan daun salam pada bagian sklerenkim (salam-sklerenkim) sebanyak masing-masing 9 gambar yang termasuk ke dalam tingkatan kedua terbawah (diantara 8-16 gambar). Akan tetapi, apabila dilihat dari hasil prediksinya, maka model berhasil memprediksi jenis tanaman obatnya.""")


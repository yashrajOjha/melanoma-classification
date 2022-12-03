import streamlit as st
from predictions import load_image
from PIL import Image

st.set_page_config(page_title="Melanoma Classification",page_icon=":computer:",layout="wide")
st.subheader("Hi, I'am Yash :wave:")
st.title("Melanoma Skin Cancer Detection Model")
st.write("I've designed this website to check for presence of melanoma by using a VGG16 model!")
st.write("[Find the code](https://www.kaggle.com/code/yashrajojha28/vgg16-model)")

check = False
dis = True
col1, col2 = st.columns(2)
with col1:
    st.subheader("Select a file to upload")
    uploaded_file = st.file_uploader("")
    if uploaded_file is not None:
        check = load_image(uploaded_file)
        if check == False or check==True: dis = False
        im = Image.open(uploaded_file)
        st.image(im, width=256)
        st.write("Image Uploaded Successfully")
with col2:
    st.subheader("Results :")
    if check==False and dis==False:
        st.write("No Melanoma Detected :slightly_smiling_face:")
    if check==True:
        st.write("Melanoma Detected! Visit a doctor!")
        st.write("[Read about melanoma](https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884#:~:text=Melanoma%2C%20the%20most%20serious%20type,in%20your%20nose%20or%20throat.)")
    if dis==True:
        st.write("Upload file for detection :file_folder:")

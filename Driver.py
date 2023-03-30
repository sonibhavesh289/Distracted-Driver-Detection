import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from re import L
from PIL import Image
import altair as alt
import random
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image 
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from keras.models import load_model

st.title('Distracted Driver Detection System')
"Hello, my self Bhavesh Varma Mtech Student at NITK." 

BEST_MODEL = "model/driver-03-0.81.hdf5"
model = load_model(BEST_MODEL)
tags = { "C0": "safe driving",
"C1": "texting - right",
"C2": "talking on the phone - right",
"C3": "texting - left",
"C4": "talking on the phone - left",
"C5": "operating the radio",
"C6": "drinking",
"C7": "reaching behind",
"C8": "hair and makeup",
"C9": "talking to passenger" }

add_selectbox = st.sidebar.selectbox(
    "Chosse Your Input form",
    ("Live Camera", "Image", "Recorded Video")
)

def liveCamera():
    st.subheader("Select to open and close live camera")
    camera = st.radio("",('Please Select to close','Please Select to Open Web camera'))
    if camera == 'Please Select to Open Web camera':
        picture = st.camera_input("Take a picture")
        if picture:
            st.image(picture)

def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0

def recordedVideo():
    uploaded_file = st.file_uploader("Choose a file",type=["mp4","mov"])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            bytes_data = uploaded_file.getvalue()
            st.video(bytes_data)
        vidcap = cv2.VideoCapture("uploads/"+uploaded_file.name) # load video from disk
        total_frame_count = (vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        duration_video =  float(total_frame_count) / float(fps)
        colList=[""]*3
        colList = st.columns(3) 
        colList[0].metric("Total Duration of Video in second: ",int(duration_video))
        colList[1].metric("Total Frames:",total_frame_count)
        colList[2].metric("Frame Per Seconds:",fps)
        success = True
        stats = [0]*10
        frame_count=0
        framePredict=[]
        allframes=[]
        time_skips = int(st.text_input('Enter seconds',1))*1000
        if st.button('Run Demo'):
            st.subheader("Wait Until result prepared")
            while True:
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_count*time_skips)) 
                success, frame = vidcap.read() # get next frame from video
                if success:
                    img = frame
                    img = img[50:,120:-50]
                    img = cv2.resize(img,(224,224)) 
                    img_array = np.array(img).reshape(-1,224,224,3)
                    prediction = model.predict(img_array)     # print(prediction)
                    for i in range(1):  
                        myclass=str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
                        stats[int(myclass)] += 1
                        predicted_class = 'C'+myclass
                        framePredict.append(predicted_class)
                        allframes.append(frame)
                else:
                    break
                frame_count+=1
            for i in range(0,5):
                colList=[""]*2
                colList = st.columns(2) 
                for j in range(0,2):
                    predicted_class = 'C'+str(i*2+j)
                    colList[j].metric(tags[predicted_class],stats[i*2+j])
            for i in range(0,len(allframes)):
                st.write(tags[framePredict[i]])
                st.image(allframes[i])

def image():
    uploaded_file = st.file_uploader("Upload Images", type=["mkv","jpg","jpeg"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        st.image(bytes_data)
        pil_image = Image.open(uploaded_file)      #this reads image in RGB
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)    #currently it is in BGR.
        img = cv2_image[50:,120:-50]
        img = cv2.resize(img,(224,224)) 
        img_array = np.array(img).reshape(-1,224,224,3)
        prediction = model.predict(img_array)
        for i in range(1): 
            predicted_class = 'C'+str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
            st.subheader(tags[predicted_class])

if (add_selectbox=="Live Camera"):
    liveCamera()
   
if(add_selectbox=="Recorded Video") : 
    recordedVideo()
    
if(add_selectbox=="Image"):
    image()
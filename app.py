import os
import time
import cv2
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import streamlit as st
from pathlib import Path
from playsound3 import playsound
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


#Custom transformation
class ResizePad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size # current width and height of the photo
        scale = self.size / max(w, h) #new desired scale of the photo
        new_w, new_h = int(w * scale), int(h * scale) #rescaling the width and the height
        img = F.resize(img, (new_h, new_w)) #resizing the photo according to the new scale

        pad_w = self.size - new_w # adding the needed padding
        pad_h = self.size - new_h # adding the needed padding
        left = pad_w // 2# adding the needed padding for each side
        right = pad_w - left # adding the needed padding for each side
        top = pad_h // 2 # adding the needed padding for each side
        bottom = pad_h - top# adding the needed padding for each side
        return F.pad(img, [left, top, right, bottom])# padding


#Static vatiables
BASE_DIR = Path(__file__).resolve().parent#
VIDS_DIR = BASE_DIR / "Vids to test"#Directory for vids

CUSTOM_MP = BASE_DIR / "DistractModel3.0.pth" #Distraction model directory
TRANSFER_MP = BASE_DIR / "DistractModelTransfer2.pth"#Distraction model tranfer directory


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #Where to run the code?

VAL_TF_CUSTOM = transforms.Compose([
    ResizePad(256), #resizing
    transforms.ToTensor(), #Trasforming to tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), #normalizing according to known values
])


VAL_TF_TRANSFER = transforms.Compose([
    ResizePad(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])





#importing custom models
class CNN_Distract(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            #first convection layer, from 3 channels to 32, then pooling the data to shrink dimentions (128)

            nn.Conv2d(32,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            #Second convection layer, from 32 channels to 64, then pooling the data to shrink dimentions (64)

            nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            #Third convection layer, from 64 channels to 128, then pooling the data to shrink dimentions (32)

            nn.AdaptiveAvgPool2d(1), #pooling to 128x1x1
            nn.Flatten(),#flattening to 128 
            nn.Dropout(0.2), #removing neurons
            nn.Linear(128, num_classes) #last linear
        )

    def forward(self, x):
        return self.step(x)






#loading the custom model
@st.cache_resource #decorater to make sure you dont need to load models each time
def load_models():
   
        # Driver detector
        person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        ).to(DEVICE).eval()

        # Custom CNN
        custom_model = CNN_Distract(10)
        custom_model.load_state_dict(torch.load(CUSTOM_MP, map_location=DEVICE))
        custom_model.to(DEVICE).eval()

        # Transfer model
        transfer_model = resnet18()
        feats = transfer_model.fc.in_features
        transfer_model.fc = nn.Linear(feats, 10)
        transfer_model.load_state_dict(torch.load(TRANSFER_MP, map_location=DEVICE))
        transfer_model.to(DEVICE).eval()

        return person_model, custom_model, transfer_model



#Classification function
def person_present(img_rgb, model, score_thr=0.6):
    with torch.no_grad():
        tf_img = transforms.ToTensor()(img_rgb).to(DEVICE)
        out = model([tf_img])[0]
        keep = (out["scores"] >= score_thr) & (out["labels"] == 1)
        return bool(keep.sum().item())


def classify(img_rgb, model, transforms):
    with torch.no_grad():
        if not isinstance(img_rgb, Image.Image):
            img_rgb = Image.fromarray(img_rgb)  #makes sure its a PIL object
        out = model(transforms(img_rgb).unsqueeze(0).to(DEVICE))#running frame through model
        return int(out.argmax(dim=1).item())#getting prediction


def getlabel(pred):
    if (pred == 0):
        return "Drinking"
    elif (pred == 1):
        return "doing hair and makeup"
    elif (pred == 2):
        return "using radio"
    elif (pred == 3):
        return "Reaching behind"
    elif (pred == 4):
        return "Driving safely"
    elif (pred == 5 or pred == 6):
        return "Using phone"
    elif (pred == 7):
        return "Talking to passenger"
    else:
        return "Texting"





def main():
    st.set_page_config(layout="wide")
    st.title("Driver Monitoring System")

    #Adding the sidebar to choose settings
    with st.sidebar:
        st.header("Controls")

        video_files =[p for p in VIDS_DIR.iterdir()] #adding video files

        input_source = st.selectbox(
            "Input Source",
            ["Webcam"] + [v.name for v in video_files] # adding video sources
        )

        model_choice = st.radio(
            "Model",
            ["Custom CNN", "Transfer (ResNet18)"], #adding model choices
            horizontal=True
        )

        start = st.button("Start", type="primary")
        stop = st.button("Stop")



    #adding running and configuring start and stop buttons
    if "running" not in st.session_state:
        st.session_state.running = False

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False



    #adding empty box for video and status
    video_box = st.empty()



    #loading the models
    person_model, custom_model, transfer_model = load_models()
    if model_choice == "Custom CNN": #using correct model
        model = custom_model
    else:
        model = transfer_model


    if model_choice == "Custom CNN":#using correct transforms
        tf = VAL_TF_CUSTOM
    else:
        tf = VAL_TF_TRANSFER




    # Opening the video source
    if input_source == "Webcam":
        capture = cv2.VideoCapture(0) #connecting to webcam
    else:
        capture = cv2.VideoCapture(str(VIDS_DIR / input_source)) #connection to relevant chosen video



    #Main loop
    last_prediction = 4
    while st.session_state.running:
        #getting the frame
        ret, frame_bgr = capture.read()
        if not ret:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #correcting to the rgb trained on


        #getting the prediction
        if person_present(frame_rgb, person_model):

            prediction = classify(frame_rgb, model, tf) #gettig prediction

            if (prediction != 4 and last_prediction != 4):
                playsound(os.path.join(BASE_DIR, "Sounds","beep.mp3")) #playing sound
            
            label = f"{getlabel(prediction)}" #showing label
            last_prediction = prediction
        else:
            label = "NO DRIVER"
            last_prediction = 4


        #displaying
        cv2.putText(frame_bgr, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)#displaying in green
        video_box.image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            width='stretch'
        )
        time.sleep(0.03) 


    capture.release()

if __name__ == "__main__":
    main()

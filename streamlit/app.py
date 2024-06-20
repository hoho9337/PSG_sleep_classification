import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import time
from diffusers import AutoPipelineForText2Image
import os
from model import CNNLSTMModel
from datetime import datetime

# Paths
font_path = 'NanumGothicBold.ttf'
encoding_file = 'encodings.json'

# prompt
def prompt(ss,eeg, ecg, emg, eog, subject):
  if ss : # 이미지 종류 ex) photo, paint,
    type_ = 'photography'
  else:
    type_ = 'illustration'

  if eeg :
    style = 'impressionism'
  else:
    style = 'realism'

  if ecg < 60:
    color = 'red-colored high chroma'
  elif ecg >= 60 and ecg <= 100:
    color = 'green-colored high chroma'
  else:
    color = 'blue-colored low chroma'

  if emg :
    additional = 'dystopian'
  else:
    additional = 'utopian'

  if eog :
    resolution = 'high resolution'
  else:
    resolution = 'low resolution'

  sentence = f'{style} {type_} of {subject}, {resolution}, overall color is {color}, {additional}'

  return sentence

# sdxl-turbo
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

def save_image(image, image_name, base_path):
    #output_dir = os.path.join(os.getcwd(), "output")
    #image_dir = os.path.join(output_dir)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    image_file = os.path.join(base_path, image_name)
    image.save(image_file, format="JPEG")
    
    return image_file
  
def create_timestamped_folder(base_path="."):
    # 현재 날짜와 시간을 포맷된 문자열로 가져오기
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 새 폴더 경로 생성
    new_folder_path = os.path.join(base_path, timestamp)
    
    # 폴더 생성
    os.makedirs(new_folder_path, exist_ok=True)
    
    return new_folder_path

# Main Streamlit app function
def main():

    st.title("🔮 당신의 꿈을 그려드립니다 🔮")
    
    # Model loading
    num_channels = 8
    num_filters = 32
    lstm_hidden_size = 64
    num_classes = 5
    
    model_save_path = './model_pth/best_model2024-06-03.pth'
    model = CNNLSTMModel(num_channels, num_filters, lstm_hidden_size, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()
    
    form_key_suffix = 0
    form = None
    
    # 본인 ID 입력
    #id = st.text_input('ID를 입력하세요')
    #fname = st.text_input('원하는 키워드를 입력하세요')
    form = st.form(f"register_form_{form_key_suffix}")
    id = form.text_input('ID를 입력하세요')
    subject = form.text_input('원하는 키워드를 입력하세요')
    type = form.text_input('원하는 이미지 종류를 입력하세요(사진, 그림)')
    submit = form.form_submit_button("제출")
    
    if submit:
      model.eval()
      input_data = torch.load(f'tensor_one/data_{id}.pt').type(torch.float32)
      input_data = input_data.unsqueeze(0)
      input_data = input_data.to(DEVICE)
      label = torch.load(f'tensor_one/label_{id}.pt')
    
      with torch.no_grad():
          output = model(input_data)
        
      predicted = torch.argmax(output, dim=2)
      predicted = predicted[0].cpu()
      predicted = predicted.numpy()
    
      label = label.cpu()
      label = label.numpy()
      
      with st.spinner('최대 1분이 소요됩니다...'):
        image_dir = create_timestamped_folder('output')
        # draw a picture
        for label in label:
          if label == 3:
              #data_30 = input_data[label]
              # REM 단계의 data_30(30초짜리 데이터)에 대해 특성추출 -> 각 신호별 임계값 추출
            
              if type == '사진':
                ss = 'photography'
              elif type == '그림':
                ss = 'illustration'
                
              eeg = 'impressionism'
              ecg = 'green-colored high chroma'
              emg = 'utopian'
              eog = 'low resolution'
              #prompt(ss, eeg, ecg, emg, eog, subject)
              prompt = f'{eeg} {ss} of {subject}, {eog}, overall color is {ecg}, {emg}'
              if prompt is not None:
                  image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
                  
                  images = os.listdir(image_dir)
                  idx = int(len(images))
                  image_name = f'{subject}_img{idx}.jpg'
                  save_image(image, image_name, image_dir)
      st.success(f'{len(os.listdir(image_dir))} images are saved')
      
      image_placeholder = st.empty()
      for image in images:
        with image_placeholder.container():
          image_path = os.path.join(image_dir, image)
          img = Image.open(image_path)
          st.image(img) # 파일명 보고 싶으면 괄호 안에 , caption=image_path 추가
          time.sleep(1)
          
      # 모든 이미지가 표시된 후 마지막 이미지를 계속 유지
      image_placeholder.image(Image.open(images[-1])) # 파일명 보고 싶으면 괄호 안에 , caption=images[-1] 추가
    

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
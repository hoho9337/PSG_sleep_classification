# 멀티모달 생체신호를 이용한 딥러닝 기반 수면 단계 분류 🛌🏻
- 🎉 2024 한국 종합 소프트웨어 학술대회 (학부생/주니어 부문) 장려상

본 프로젝트는 수면 뇌파 데이터를 포함한 다양한 생체신호 데이터를 종합적으로 분석하고, 이를 기반으로 수면 상태를 시각화한다. 이미지나 비디오를 보는 중 측정하여 label이 있는 뇌파를 기반으로 이미지를 재구성하는 연구는 많이 진행되었으나, 우리 데이터는 수면 생체 신호를 이용하여 label이 없는 상태에서 데이터 특성을 바탕으로 이미지를 생성한다. 이는 뇌졸중 환자와 같이 의사소통이 불가능한 사람들의 의사 파악 등에 활용할 수 있다는 점에서 의의가 있다.
우리 프로젝트는 먼저 EEG, EMG, EOG, ECG 신호를 CNN+LSTM 모델을 통해 수면단계를 분류하고, 각 신호에 대하여 feature extraction을 진행한다. 이후 생성형 AI 기술을 사용하여, 추출한 특성들을 바탕으로 시각화된 이미지를 산출한다.

## 요약
멀티모달 생체신호를 활용한 딥러닝 기반 수면 단계 분류 모델은 수면 질 평가 및 관련 질환 진단에 있어서 핵심적인 역할을 수행한다. 본 연구에서는 Haaglanden Medisch Centrum(HMC)의 공개 데이터셋을 기반으로 다양한 생체 신호를 융합하여 수면 단계 분류 성능을 향상시키는 CNN+LSTM 모델을 제안한다. 151명PSG(Polysomnography) 데이터의 EEG, EOG, ECG, EMG 신호를 멀티모달 형태로 결합하여, 각 신호가 지닌 고유한 특징을 효과적으로 활용하고자 하였다. CNN을 통해 시공간적 특징을 추출하고, LSTM을 통해 시간적 의존성을 학습하여 수면 단계 분류하였다. 실험 결과, 멀티모달 생체신호 데이터를 활용한 모델은 단일 신호 기반 모델 대비 정확도 평균 13.05%(F1-score = 0.16) 향상되는 성능을 보였으며, 멀티모달 데이터 융합의 효과를 입증하였다. 본 연구는 정밀한 수면 분석을 위한 새로운 가능성을 제시하며, 향후 수면 질환 진단 및 치료에 기여할 수 있을 것으로 기대된다.

# Data
Haaglanden Medisch Centrum sleep staging database

URL: https://physionet.org/content/hmc-sleep-staging/1.1

# Data Preprocessing
Band-Pass Filtering, Notch Filtering, Wavelet denoising, Resampling, Z-score normalization('./_Preprocessing.ipynb')

<img width="931" alt="Screenshot 2024-06-18 at 14 35 52" src="https://github.com/hoho9337/PSG_sleep_classification/assets/97961767/e96d5673-7713-49b3-b95d-cec7eebbaed0">


# Model
2 layers of 1D-CNN + LSTM ('./tensor_to_CNN_LSTM.ipynb')

Preprocessed .edf data was processed into pytorch tensors ('./edf_to_tensor.ipynb')

![model](https://github.com/hoho9337/sleep-EOG-analysis/assets/97961767/5ccb71ae-4735-4268-a8e3-ad99b4fbf1f8)

# 논문 본문
![PSG_SleepClassification](https://github.com/user-attachments/assets/adfabeb1-6ede-4277-b714-315cc81e6653)
![PSG_SleepClassification2](https://github.com/user-attachments/assets/a1bdb38f-0929-45c2-b26d-998860d0bc2d)
![PSG_SleepClassification3](https://github.com/user-attachments/assets/85138ecc-46c1-4b78-8c0f-ac7370d51144)


# 결과 (이미지 생성 부분)
![image](https://github.com/yyaaoonngg/PSG_sleep_classification/assets/97872145/03267157-657b-4e90-9773-d5f38d192acb) <br>
[서비스 화면] 최종 코드를 돌리면 생성되는 서비스 화면이다. 사용자는 보고 싶은 꿈의 내용을 입력한다.<br>
![image](https://github.com/yyaaoonngg/PSG_sleep_classification/assets/97872145/2b2c0012-af27-451d-ad5d-17f3842dda3f) <br>
[이미지 출력 결과] 한 사람의 각 램수면 단계에서의 이미지를 보여준다.

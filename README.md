# Facial Landmark Detection

### 1.  Quick start

（1）Clone the project

````bash
git clone https://github.com/HRNet/HRNet-Facial-Landmark-Detection.git
````

（2）Install dependencies

````bash 
pip3 install -r requirements.txt
````

（3） download pre-trained model and test

~~~python
python3 camera.py--cfg <CONFIG-FILE> --model-file <MODEL WEIGHT> 
# example:
python3 camera.py  --cfg experiments/face_landmark_detection_wflw_shufflenet_large.yaml  --model-file pretrained/shufflenet_plus.pth
~~~

### 2. Datasets

- Download the annotations files from:
  Google Drive: https://drive.google.com/file/d/1W8p0MWmUtWxH1B2LsImtg3JVO-o5AU9u/view?usp=sharing

  Baiduyu Link: https://pan.baidu.com/s/10l70jaoWf5ls4t6HMoFk1w     Access Code: 6ipb 

- Download images (WFLW) from official websites and then put them into `images` folder for each dataset.

Your `data` directory should look like this:

````shell
.
└──data
    └── wflw
        ├── face_landmarks_wflw_test_blur.csv
        ├── face_landmarks_wflw_test.csv
        ├── face_landmarks_wflw_test_expression.csv
        ├── face_landmarks_wflw_test_illumination.csv
        ├── face_landmarks_wflw_test_largepose.csv
        ├── face_landmarks_wflw_test_makeup.csv
        ├── face_landmarks_wflw_test_occlusion.csv
        ├── face_landmarks_wflw_train.csv
        └── images

2 directories, 8 files
````

### 3. Training

````bash
python train.py --cfg <CONFIG-FILE>
# example:
python3 train.py --cfg experiments/face_alignment_wflw_hrnet_w18.yaml
````

### 4. benchmark

##### WFLW

|       NME       | model_size | *test* | *pose* | *illumination* | *occlution* | *blur* | *makeup* | *expression* |
| :-------------: | ---------- | :----: | :----: | :------------: | :---------: | :----: | :------: | :----------: |
| shufflenet_plus | 13.8M      |  4.79  |  8.56  |      4.73      |    5.80     |  5.47  |   4.77   |     5.15     |
|      HRNet      | 39.2M      |  4.60  |  7.86  |      4.57      |    5.42     |  5.36  |   4.26   |     4.78     |

### 5. project structure

~~~shell
.
├── data
│   └── wflw
│       ├── face_landmarks_wflw_test_blur.csv
│       ├── ...
│       ├── face_landmarks_wflw_train.csv
│       └── images
├── experiments
│   └── face_landmark_detection_wflw_shufflenet_large.yaml
├── output
│   ├── log
│   │   └── WFLW
│   └── WFLW
│       └── face_landmark_detection_wflw_shufflenet_large
├── README.md
├── requirements.txt
├── src
│   ├── datasets.py
│   ├── __init__.py
│   ├── loss.py
│   ├── models
│   │   ├── hrnet.py
│   │   ├── __init__.py
│   │   ├── shufflenet_bak.py
│   │   ├── shufflenet.py
│   │   └── utils.py
│   ├── transforms.py
│   └── utils.py
├── test.py
└── train.py  

13 directories, 28 files
~~~

### 6. TBD

- [ ] face pose weighted
- [ ] heatmap 
- [ ] graph network
- [ ] model、dataset、loss
- [ ] deployment
- [ ] video stable

### 7. Reference

https://github.com/HRNet/HRNet-Facial-Landmark-Detection
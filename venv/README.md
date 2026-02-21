It is recommended to use a virtual environment to manage project dependencies separately.
## 1. Navigate to Project Folder

cd music_genre_classification

## 2. Create Virtual Environment

python -m venv venv
**This creates a folder named venv/**

## 3. Activate Virtual Environment

venv\Scripts\Activate.ps1
**After activation, there will be (venv) in the terminal**

## 4. Install Dependencies

pip install tensorflow==2.13.0
pip install numpy--1.24.3
pip install  librosa==0.10.1
pip install matplotlib==3.7.2
pip install opencv-python==4.8.0.76
pip install streamlit==1.28.0
pip install scikit-learn==1.3.0

## 5. Run the Application

streamlit run ./app/app.py

## 6. Deactivate Environment

deactivate


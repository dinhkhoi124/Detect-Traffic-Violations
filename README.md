# Detect Traffic Violations: Helmet & License Plate Detection with YOLOv11
Detect-Traffic-Violations is an advanced computer vision system designed to automate road safety surveillance. The project focuses on real-time detection of motorcyclists without helmets and license plate identification using a YOLOv11 model fine-tuned on a custom real-world dataset.

## Key Features
- **High-Accuracy Object Detection:** Utilizes the YOLOv11n architecture to achieve an optimal balance between inference speed and detection precision.

- **Performance Metrics:** The model achieved impressive results during evaluation:

  - **mAP@50:** Reached 0.942, ensuring highly reliable detection.

  - **Precision: 0.916 and Recall: 0.887**, effectively minimizing false positives and missed detections.

- **Multi-Class Identification:** Precisely classifies three distinct categories: With Helmet, Without Helmet, and License Plate.

- **Interactive Interface:** Features a web-based UI powered by Gradio, allowing users to perform visual testing via image uploads or video streams.

## System Architecture

The system processes data through a structured pipeline:

**1. Perception Layer (YOLOv11 Pipeline)**
- **Data Augmentation:** Employs techniques such as **Mosaic, Mixup, and HSV augmentation** to enhance the model's generalization capabilities.

- **Inference:** The model processes input frames, performs bounding box predictions, and assigns violation labels based on a defined Confidence Threshold.

**2. Application Layer (Gradio UI)**
- Processes user-submitted media and returns low-latency visualized results, making it suitable for Edge AI deployment on devices like the NVIDIA Jetson.

## Project Structure
```
.
├── app.py                  
├── requirements.txt       
├── .gitignore              
├── demo/                   
│   ├── img_demo.jpg
│   └── screenshot_...png
├── notebook_model/         
│   └── helmet-v3.ipynb     
└── weights/                
    └── best.pt             
```

- ```app.py```: The main script to launch the web interface.

- ```notebook_model/```: Documentation of the model development and training process.

- ```weights/best.pt```: The core fine-tuned model for traffic violation detection.

## Installation

**Clone the repository**
```
git clone https://github.com/dinhkhoi124/Detect-Traffic-Violations.git
cd your-repository-name
```

**Create and activate virtual environment**
```
python -m venv venv
```

**Windows:**
```
.\venv\Scripts\activate 
```

**Linux/Mac:**
```
source venv/bin/activate
```

**Install dependencies:**:
```
pip install -r requirements.txt
```
⚠️ Recommended: Python ≥ 3.9 and an NVIDIA GPU for optimal inference speed.

## Usage
Ensure your model weights are placed in the weights/ folder.

Run the Gradio application:
```
python app.py
```
Access the local URL provided in the terminal (usually http://127.0.0.1:7860).

## Author

Dinh Van Anh Khoi 

🎓 AI Engineer (Final-year student) 

💡 Interests: Computer Vision, Edge AI, Smart City Infrastructure, and Robust Deep Learning in Adverse Environments.

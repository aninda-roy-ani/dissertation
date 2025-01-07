Project Title: Towards Automated Dermatological Diagnostics: An Integration of ResNet and MobileNet for Skin Disease Classification 
This project demonstrates the fine-tuning and evaluation of pre-trained deep learning models (MobileNetV2 and ResNet50) on the HAM10000 dataset for skin disease classification.

Dataset Structure:
To run the code, you need the HAM10000 dataset, (link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), organized as follows:
1.	Create a folder named output_data_ham in the root directory of the project.
2.	Inside output_data_ham, create the following subfolders:
  o	train/ - Contains training images for each class.
  o	validate/ - Contains validation images for each class.
  o	test/ - Contains testing images for each class.

The dataset should have the following directory structure:
bash
Copy code
output_data_ham/
├── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── validate/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── test/
    ├── akiec/
    ├── bcc/
    ├── bkl/
    ├── df/
    ├── mel/
    ├── nv/
    └── vasc/
  	
Make sure that each class folder contains the corresponding images for that class.

Requirements:
Install the following dependencies before running the code:
•	Python 3.7 or later
•	TensorFlow 2.8 or later
•	NumPy
•	Matplotlib
•	Scikit-learn
You can install the dependencies using:
bash
Copy code
pip install -r requirements.txt
(Ensure the requirements.txt file includes all the dependencies above.)

Files and Their Purpose:
data_loader.py
This script handles loading and preprocessing the dataset. It applies data augmentation and scaling for better model generalization. The data is split into train, validation, and test sets.
model_builder.py
Contains code to build and fine-tune MobileNetV2 and ResNet50 models. The models are pre-trained on ImageNet and adapted to classify the 7 classes in the HAM10000 dataset.
trainer.py
Manages the training and fine-tuning process. Includes callbacks such as early stopping and model checkpointing to save the best-performing model during training.
main.py
The main script for running the entire workflow. It orchestrates data loading, model training, fine-tuning, and evaluation.
test.py
Evaluates the trained models on the test dataset and generates performance metrics like accuracy.

Running the Project:
1.	Clone this repository:
bash
Copy code
git clone https://github.com/aninda-roy-ani/dissertation.git
cd dissertation
2.	Place the HAM10000 dataset in the required directory structure (output_data_ham as described above).
3.	Run the main script:
bash
Copy code
python main.py
This script will:
•	Load the HAM10000 dataset.
•	Train and fine-tune MobileNetV2 and ResNet50 models.
•	Save the fine-tuned models in the models/ directory.
•	Evaluate the models on the test dataset.
Results:
•	MobileNetV2 achieved 88% accuracy after fine-tuning.
•	ResNet50 achieved 92% accuracy, surpassing previous benchmarks.
Notes:
•	The fine-tuned models are saved in the models/ folder with the format <model_name>_best.h5.
•	Modify the EPOCHS, BATCH_SIZE, or LEARNING_RATE in main.py to experiment with hyperparameters.

License:
This repository is licensed under the GPL-3.0 License. See the LICENSE file for details.
Acknowledgments
•	HAM10000 Dataset: Kaggle
•	MobileNetV2 and ResNet50: TensorFlow Models

Feel free to contribute to this project or raise issues in the repository.


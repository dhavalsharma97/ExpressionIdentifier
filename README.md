# ExpressionIdentifier

<h2>Instructions to run the code:</h2><br>
<b>Download the dataset:</b><br>
<b>Link to the dataset -</b> https://www.kaggle.com/deadskull7/fer2013<br>
Create a folder with the name "data" in the root directory of the project. Download and place the Fer2013 dataset in this data folder.<br>
Place the shuffle_dataset.py file inside the "data" folder.<br>
Run the shuffle_dataset.py in order to shuffle the dataset as many times as you want.<br>

<b>[2] Install the required dependencies:</b><br>
<b>Numpy -</b> pip install numpy<br>
<b>ArgParse -</b> pip install argparse<br>
<b>Pandas -</b> pip install pandas<br>
<b>Hyperopt -</b> pip install hyperopt<br>
<b>Scikit Learn -</b> pip install sklearn<br>
<b>Scikit-Image -</b> pip install scikit-image<br>
<b>Dlib -</b> pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf<br>
<b>OpenCV -</b> pip install opencv-python<br>
<b>Tensorflow -</b> https://www.tensorflow.org/install<br>


<b>[3] Run the various models:</b><br>
<b>(a) Convolution Neural Network:</b><br>
NOTE: You need to have your camera working in order for this to work.<br>
<b>1. To run the code -</b><br> Type "python3 main.py" in the command prompt. This command will open a window displaying the video taken from your web camera. Press "Space" to capture the frame and get the required expression.<br>
<b>2. To train the model -</b><br> Update the "mode" (main.py) from "demo" to "train". Then, type "python3 main.py" in the command prompt.<br>


<b>NOTE:</b> The convert_dataset files in the below models limits the total samples to 500 for each labels. If you want to train and test the model on the whole dataset, please do the following inside the "convert_dataset.py":<br>
<b>Replace the line -</b> "if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:"<br>
<b>with this line -</b> "if labels[i] in SELECTED_LABELS:"<br>


<b>(b) Support Vector Machine:</b><br>
<b>1. Download the model and place it in the SVM folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>
<b>5. Customize the training parameters:</b><br>
Feel free to change the values of the parameters in the "parameters.py" file accordingly.</b><br>
<b>6. Find the best hyperparameters (using hyperopt):</b><br>
<b>Command Prompt -</b> python optimize_parameters.py --max_evals=15<br>


<b>(c) Decision Tree:</b><br>
<b>1. Download the model and place it in the Decision Tree folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>


<b>(d) K Nearest Neighbors:</b><br>
<b>1. Download the model and place it in the K Nearest Neighbors folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>


<b>(e) Logistic Regression:</b><br>
<b>1. Download the model and place it in the Logistic Regression folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>


<b>(f) Naive Bayes:</b><br>
<b>1. Download the model and place it in the Naive Bayes folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>


<b>(g) Random Forest:</b><br>
<b>1. Download the model and place it in the Random Forest folder:</b><br>
Dlib Shape Predictor Model - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
<b>2. Convert the dataset to extract Face Landmarks and HOG Features:</b><br>
<b>Command Prompt -</b> python convert_dataset.py<br>
<b>3. Train the model:</b><br>
<b>Command Prompt -</b> python train.py --train=yes<br>
<b>4. Evaluate the model:</b><br>
<b>Command Prompt -</b> python train.py --evaluate=yes<br>

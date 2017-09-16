# 1.Generate Training Data

~/face2face-demo/
python generate_train_data.py --file Trump.mp4 --num 400 --landmark-model shape_predictor_68_face_landmarks.dat

Input:

file is the name of the video file from which you want to create the data set.
num is the number of train data to be created.
landmark-model is the facial landmark model that is used to detect the landmarks.

Output:

Two folders original and landmarks will be created.


# 2.Train Model

1)Move the original and landmarks folder into the pix2pix-tensorflow folder
~/face2face/
mv face2face-demo/landmarks face2face-demo/original pix2pix-tensorflow/photos

2)Go into the pix2pix-tensorflow folder
cd pix2pix-tensorflow/

3)Resize original images
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --output_dir photos/original_resized
  
4)Resize landmark images
python tools/process.py \
  --input_dir photos/landmarks \
  --operation resize \
  --output_dir photos/landmarks_resized
  
5)Combine both resized original and landmark images
python tools/process.py \
  --input_dir photos/landmarks_resized \
  --b_dir photos/original_resized \
  --operation combine \
  --output_dir photos/combined
  
6)Split into train/val set
python tools/split.py \
  --dir photos/combined
  
7)Train the model on the data
python pix2pix.py \
  --mode train \
  --output_dir face2face-model \
  --max_epochs 200 \
  --input_dir photos/combined/train \
  --which_direction AtoB
  
  
  
  
# 3.Export Model
  
 
1) we need to reduce the trained model so that we can use an image tensor as input:
~/face2face-demo/
python reduce_model.py --model-input face2face-model --model-output face2face-reduced-model


Input:

model-input is the model folder to be imported.
model-output is the model (reduced) folder to be exported.

In this case face2face-model=../pix2pix-tensorflow/face2face-model

Output:

It returns a reduced model with less weights file size than the original model.

2)we freeze the reduced model to a single file.

python freeze_model.py --model-folder face2face-reduced-model

Input:

model-folder is the model folder of the reduced model.

Output:

It returns a frozen model file frozen_model.pb in the model folder.



# 4.Run Demo

python run_webcam.py --source 0 --show 0 --landmark-model shape_predictor_68_face_landmarks.dat --tf-model face2face-reduced-model/frozen_model.pb

Input:

source is the device index of the camera (default=0).
show is an option to either display the normal input (0) or the facial landmark (1) alongside the generated image (default=0).
landmark-model is the facial landmark model that is used to detect the landmarks.
tf-model is the frozen model file.

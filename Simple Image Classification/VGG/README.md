# Supervised Learning Classification Task (VGG16)


### Prerequisites
* python==3.7
* numpy==1.21.6
* opencv_python==4.2.0.34
* Pillow==9.2.0
* pytesseract==0.3.10
* scikit_learn==1.0.2
* tifffile==2020.7.24
* torch==1.11.0+cu113
* torchvision==0.12.0+cu113
* torchvision==0.6.1+cu92


### Training
* CMD:
 $ python model.py

* Anaconda Prompt:
 $ conda activate
 $ conda activate pytorch
 $ python model.py


### Results
 * Trained in 10 epoches
   Epoch 1: Train Loss = 1.1267834788277036, Val Loss = 0.8005639836192131
   Epoch 2: Train Loss = 0.8663532336552938, Val Loss = 0.7303657867014408
   Epoch 3: Train Loss = 0.8296227029391697, Val Loss = 0.6702460385859013
   Epoch 4: Train Loss = 0.7928601664210123, Val Loss = 0.657683476805687
   Epoch 5: Train Loss = 0.7545939717027876, Val Loss = 0.6717860884964466
   Epoch 6: Train Loss = 0.750568677035589, Val Loss = 0.6189089901745319
   Epoch 7: Train Loss = 0.7390636853755467, Val Loss = 0.6690583750605583
   Epoch 8: Train Loss = 0.7383055005754743, Val Loss = 0.6317843608558178
   Epoch 9: Train Loss = 0.7181661251991515, Val Loss = 0.6136190667748451
   Epoch 10: Train Loss = 0.7162374768938337, Val Loss = 0.6287502981722355

* Accuracy: 74.00%
* Precision: 76.46%
* Recall: 75.02%
* F1: 73.86%
* Confusion Matrix:
  [[48  1  1  6  0]
   [ 3 41  0  2  0]
   [ 1  2 40  3  6]
   [ 6  0  0 31  2]
   [ 4  2  2 24 25]]


# Supervised Learning Classification Task (ResNet50)


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
    Epoch 1: Train Loss = 0.8442356293163602, Val Loss = 0.9874075651168823
    Epoch 2: Train Loss = 0.5763707300500264, Val Loss = 1.1773799508810043
    Epoch 3: Train Loss = 0.5340836925639046, Val Loss = 0.5385082848370075
    Epoch 4: Train Loss = 0.4044802349711221, Val Loss = 3.622736304998398
    Epoch 5: Train Loss = 0.3678293295559429, Val Loss = 0.7457750663161278
    Epoch 6: Train Loss = 0.35368887595241033, Val Loss = 0.9906599968671799
    Epoch 7: Train Loss = 0.3397943294710583, Val Loss = 0.6582075990736485
    Epoch 8: Train Loss = 0.2510034493983738, Val Loss = 0.7693624459207058
    Epoch 9: Train Loss = 0.2077201014709851, Val Loss = 1.7705445140600204
    Epoch 10: Train Loss = 0.15799945249917016, Val Loss = 1.0987700074911118

* Accuracy: 71.08%
* Precision: 80.00%
* Recall: 70.84%
* F1: 71.47%
* Confusion Matrix:
  [[19  0  2  3 20]
   [ 3 44  2  0  6]
   [ 0  0 37  0  9]
   [ 0  0  2 33 21]
   [ 0  0  3  1 44]]

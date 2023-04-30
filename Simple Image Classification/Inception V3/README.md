# Supervised Learning Classification Task (Inception V3)


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
    Epoch 1: Train Loss = 3.1668681284737965, Val Loss = 1.3805721700191498
    Epoch 2: Train Loss = 1.2758764671900915, Val Loss = 1.1373096853494644
    Epoch 3: Train Loss = 1.1190095127574982, Val Loss = 1.045312061905861
    Epoch 4: Train Loss = 1.0256588733385479, Val Loss = 0.9953677281737328
    Epoch 5: Train Loss = 1.003524591052343, Val Loss = 0.9548404142260551
    Epoch 6: Train Loss = 0.9461814155654301, Val Loss = 0.903588093817234
    Epoch 7: Train Loss = 0.946122185578422, Val Loss = 0.9082484394311905
    Epoch 8: Train Loss = 0.918938159942627, Val Loss = 0.8998380899429321
    Epoch 9: Train Loss = 0.9073408936697339, Val Loss = 0.8172715753316879
    Epoch 10: Train Loss = 0.8834851620689271, Val Loss = 0.8829734921455383

* Accuracy: 73.20%
* Precision: 73.38%
* Recall: 73.39%
* F1: 72.90%
* Confusion Matrix: 
  [[36 10  3  2  2]
   [ 5 39  3  1  0]
   [ 2  0 43  3  1]
   [ 3  3  1 37  6]
   [ 3  5  5  9 28]]

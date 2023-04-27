# Supervised Learning Classification Task (CNN)


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
    Epoch 1: Train Loss = 12.396170453419762, Val Loss = 2.22947259247303
    Epoch 2: Train Loss = 1.1212856480999598, Val Loss = 0.9337271712720394
    Epoch 3: Train Loss = 0.6805746711435772, Val Loss = 0.7494149841368198
    Epoch 4: Train Loss = 0.5006039845091956, Val Loss = 0.6943503506481647
    Epoch 5: Train Loss = 0.3154465774340289, Val Loss = 0.6758411265909672
    Epoch 6: Train Loss = 0.21613934287239636, Val Loss = 0.7889119181782007
    Epoch 7: Train Loss = 0.13697257797632897, Val Loss = 0.767864853143692
    Epoch 8: Train Loss = 0.15360806639572339, Val Loss = 0.9860433042049408
    Epoch 9: Train Loss = 0.1746810924202677, Val Loss = 1.0631957575678825
    Epoch 10: Train Loss = 0.3315029663684231, Val Loss = 1.149612719193101

 * Accuracy: 79.68%
 * Precision: 79.60%
 * Recall: 80.10%
 * F1: 79.29%
 * Confusion Matrix:
    [[32  1  3  0  1]
     [ 6 51  2  1  0]
     [ 2  1 43  0  2]
     [ 3  0  4 40  9]
     [ 2  5  6  3 34]]
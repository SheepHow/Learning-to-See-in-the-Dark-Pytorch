# Learning-to-See-in-the-Dark-Pytorch
This is a pytorch implementation of Learning to See in the Dark in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.

## Requirements
- Python 3.6
- Pytorch 1.3
- RawPy 0.13.1
- SciPy 1.0.0
```
pip install -r requirements.txt
```
## Dataset
Please follow the steps from the original [code](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Testing
Please download the pretrained models [here](https://drive.google.com/drive/folders/1DvDDEYhBNtlrsCvIRO5GI5ghkUGoRjDI?usp=sharing), and put them in the folds "checkpoint/Sony/" and "checkpoint/Fuji/" respectively.


1. To test Sony data, run
```
python test_Sony.py
```
By default, the result will be saved in "result_Sony/final" folder.

2. To test Fuji data, run
```
python test_Fuji.py
```
By default, the result will be saved in "result_Fuji/final" folder.

## Training
1. To train the Sony model, run
```
python train_Sony.py
```
The result and model will be saved in "result_Sony" folder by default.

2. To train the Fuji model, run
```
python train_Fuji.py
```
The result and model will be saved in "result_Fuji" folder by default.

## Citation
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

## License
MIT License

## Questions
Please contact me you have any questions. sheephow@gapp.nthu.edu.tw
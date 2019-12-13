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
Following the steps from the original [code](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Test
```
python test_Sony.py
```
By default, the code takes the model in the "checkpoint/Sony/model.pth" and the result will be saved in "./result_Sony/final" folder.

## Train
1. To train the Sony model, run
```
python train_Sony.py
```
The result will be saved in "result_Sony" folder and the model will be saved in "checkpoint/Sony" folder by default.

## Citation
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

## License
MIT License

## Questions
Please contact me you have any questions. sheephow@gapp.nthu.edu.tw
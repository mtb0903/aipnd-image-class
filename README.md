# PyTorch Image Classifier

This code was developed and submitted as project "Create Your Own Image Classifier" for Udacity's AI Programming with Python Nanodegree program.

The project consists of a **Jupyter Notebook** and a **CLI application** that can be used to train, validate and test and image classifier using PyTorch. The code supports using **vgg16**, **resnet18** and **densenet161** as pretrained models, which then get modified with an own classifier suitable to use the [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

## CLI Examples

### vgg16
```
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch vgg16 --save_dir /home/workspace/ImageClassifier --epochs 15 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 4096 2048 1024 512 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_vgg16.txt
```
[Training output](example_out/20210313-125121_training_vgg16.txt)
```
python predict.py 20210313-125121_vgg16_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_vgg16.txt
```
[Predication output](example_out/20210316-0951_predict_vgg16.txt)

### resnet18
```
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch resnet18 --save_dir /home/workspace/ImageClassifier --epochs 10 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 512 256 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_resnet18.txt
```
[Training output](example_out/20210315-092209_training_resnet18.txt)
```
python predict.py 20210315-092209_resnet18_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_resnet18.txt
```
[Predication output](example_out/20210316-0951_predict_resnet18.txt)

### densenet161
```
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch densenet161 --save_dir /home/workspace/ImageClassifier --epochs 10 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 2208 1024 512 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_densenet161.txt
```
[Training output](example_out/20210315-101225_training_densenet161.txt)
```
python predict.py 20210315-101225_densenet161_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_densenet161.txt
```
[Predication output](example_out/20210316-0952_predict_densenet161.txt)

## Dependencies

Generated for conda environment:
`conda list -e >` [conda_requirements.txt](conda_requirements.txt)


Generated for pip environment:
`pip freeze > `[pip_requirements.txt](pip_requirements.txt)

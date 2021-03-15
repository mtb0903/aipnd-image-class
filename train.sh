# Examples trainings

# vgg16
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch vgg16 --save_dir /home/workspace/ImageClassifier --epochs 15 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 4096 2048 1024 512 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_vgg16.txt

# resnet18
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch resnet18 --save_dir /home/workspace/ImageClassifier --epochs 10 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 512 256 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_resnet18.txt

# densenet161
python train.py data_dir=/home/workspace/ImageClassifier/flowers --arch densenet161 --save_dir /home/workspace/ImageClassifier --epochs 10 --learning_rate 0.0001 --dropout_p 0.5 --hidden_units 2208 1024 512 --gpu 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_training_densenet161.txt
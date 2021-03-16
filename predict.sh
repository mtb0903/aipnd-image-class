# Examples prediction for single image file using existing checkpoints

# Using vgg16 checkpoint
python predict.py 20210313-125121_vgg16_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_vgg16.txt

# Using resnet18 checkpoint
python predict.py 20210315-092209_resnet18_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_resnet18.txt

# Using densenet161 checkpoint
python predict.py 20210315-101225_densenet161_checkpoint.pth -f flowers/test/1/image_06743.jpg -g 2>&1 | tee -a  $(date '+%Y%m%d-%H%M')_predict_densenet161.txt


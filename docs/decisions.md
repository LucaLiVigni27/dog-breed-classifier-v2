# Decisions Log

## 09-2026 - Framework: PyTorch + timm
- timm gives easy acess to pretrained ConvNeXt,
EfficientNet, and ViT models. V1 used TensorFlow/Keras.

## 09-2026 - V1 baseline is a reconstruction
- Only v1's data-loading script was recovered. The baseline will be a small from-scratch CNN matching v1's known settings (200x200, [0, 1] scaling, RMSprop, no augmentation)

## 09-2026 - Data source: Stanford Dogs subset
- The dataset is a 12-breed subset of the Stanford Dogs dataset (2011),
 roughly cleaned during v1. 1,835 images total, 86-224 per breed.
- Stanford Dogs is derived from ImageNet. Transfer-learning results may be slightly optimistic as a result.
- http://vision.stanford.edu/aditya86/ImageNetDogs/

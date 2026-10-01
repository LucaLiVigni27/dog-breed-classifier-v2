# Decisions Log

## 09-2026 - Framework: PyTorch + timm
- timm gives easy acess to pretrained ConvNeXt,
EfficientNet, and ViT models. V1 used TensorFlow/Keras.

## 09-2026 - V1 baseline is a reconstruction
- Only v1's data-loading script was recovered. The baseline will be a small from-scratch CNN matching v1's known settings (200x200, [0, 1] scaling, RMSprop, no augmentation)

## 09-2026 - Data source: Stanford Dogs subset
- Stanford Dogs is derived from ImageNet. Transfer-learning results may be slightly optimistic as a result.
- http://vision.stanford.edu/aditya86/ImageNetDogs/

## 09-2026 - Use the full Stanford Dogs images for our 12 breeds, no manual removals
- v1's curated subset (1,835 images) was filtered unevenly and without documented
  criteria. No v1 results survived, so there is nothing to keep comparable.
- New dataset: every Stanford Dogs image of the 12 breeds from v1, rebuilt with
  `python -m dogbreeds.build_dataset`.
- No manual removals: reviewing only part of the data would apply uneven standards
  again. Known issues are documented in the data card instead.
- Mislabels will be looked for after the first training using one rule for the whole dataset.

## 09-2026 - Scope: the 12 breeds
- Same 12 breeds as v1, since v2 is a rebuild of that project.
- They include deliberate look-alike groups (Maltese / Shih-Tzu / Yorkshire Terrier,
  Basset / Bloodhound, Siberian Husky / German Shepherd).
- Small enough to train on a free Colab GPU. More Stanford breeds can be added later
  by extending BREED_WNIDS in src/dogbreeds/breeds.py.

## 09-2026 - Model output scope
- The model is a closed-set classifier over the 12 breeds. Plan: evaluate a confidence
  threshold for "not one of these" using the other Stanford breeds as unseen dogs.

# Decisions Log

## 09-2026 - Framework: PyTorch + timm
- timm gives easy access to pretrained ConvNeXt,
EfficientNet, and ViT models. V1 used TensorFlow/Keras.

## 09-2026 - V1 baseline is a reconstruction
- Only v1's data-loading script was recovered. The baseline will be a small from-scratch CNN matching v1's known settings (200x200, [0, 1] scaling, RMSprop, no augmentation)

## 09-2026 - Data source: Stanford Dogs subset
- Stanford Dogs is derived from ImageNet. Transfer-learning results may be slightly optimistic as a result.
- http://vision.stanford.edu/aditya86/ImageNetDogs/

## 09-2026 - Use the full Stanford Dogs images for our 12 breeds, no manual removals
- v1's recovered subset (1,835 images) was filtered unevenly and without documentation.
  No v1 results survived, so there is nothing to keep comparable.
- New dataset: every Stanford Dogs image of the 12 breeds from v1, rebuilt with
  `python -m dogbreeds.build_dataset`.
- No manual removals: reviewing only part of the data would apply uneven standards. Known issues are listed in the data card.
- Mislabels will be looked for after the first training using one rule for the whole dataset.

## 09-2026 - The 12 breeds
- Same 12 breeds as v1, since v2 is a rebuild of that project.
- They include deliberate look-alike groups (Maltese / Shih-Tzu / Yorkshire Terrier,
  Basset / Bloodhound, Siberian Husky / German Shepherd).
- Small enough to train on a free Colab GPU. More Stanford breeds can be added later
  by extending BREED_WN_IDS in src/dogbreeds/breeds.py.

## 09-2026 - Model output scope
- The model is a closed-set classifier over the 12 breeds. Plan: evaluate a confidence
  threshold for "not one of these" using the other Stanford breeds as unseen dogs.

## 10-2026 - Duplicates and split
- Perceptual hashing (pHash, distance <= 8) found 3 duplicate pairs (2 Bloodhound, 1 Maltese).
  Kept, but each pair stays in one split.
- Stratified 70/15/15 train/val/test split per breed, seed 42, saved to
  data/splits.csv (committed to git). Every experiment uses this file.

## 10-2026 - Pretrained weights
- Models use ImageNet-1k pretrained weights, pinned by name (convnext_tiny.fb_in1k,
efficientnet_b0.ra_in1k). timm's default convnext_tiny uses ImageNet-12k, which would
add even more overlap with Stanford Dogs

## 10-2026 - Model selection by validation loss
- First runs (val accuracy): v1 baseline 33.1%, EfficientNet-80 95.4%, ConvNeXt-Tiny 99.7%.
- ConvNeXt reached 99.1% with only the head trained (frozen ImageNet features), and its
best-accuracy checkpoint was that head-only epoch (val loss 0.90 vs 0.15 after fine-tuning).
Frozen ImageNet-1k features already separate these 12 breeds almost perfectly, which is the
ImageNet overlap issue in practice.
- At this accuracy, epochs different by a handful of images, so the best epoch is now chosen
by lowest val loss. ConvNeXt lr lowered from 3e-4 to 1e-4 (unfreezing at 3e-4 reduced val accuracy).
- Added a "fresh photos" set (not from ImageNet / Stanford Dogs) to measure performance on
genuinely new images.

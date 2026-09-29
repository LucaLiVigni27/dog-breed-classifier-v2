# V1 Notes (2024)

V1 was a team project built in Google Colab. Only the data-loading script
`v1_data_loading.py` survived; the model, training code, and results were lost.

**What it did:** TensorFlow/Keras, 200×200 images scaled to [0, 1], 80/20 train/validation split with `ImageDataGenerator`. It imported `RMSprop`, so the model was likely a small CNN trained from scratch.

**Gaps v2 fixes:** no test set, no controlled split or duplicate check, no augmentation, no transfer learning.

**V2 baseline:** a reconstructed "v1-style" CNN (200×200, RMSprop, no augmentation) trained on the v2 splits, to measure improvement against.

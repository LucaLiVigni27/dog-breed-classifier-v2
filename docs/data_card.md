# Data card

**Dataset:** every Stanford Dogs image of 12 breeds: 2,158 RGB JPEGs.
**Source:** [Stanford Dogs]
(http://vision.stanford.edu/aditya86/ImageNetDogs/)
(Khosla et al., *Novel Dataset for Fine-Grained Image Categoriation*,
FGVC workshop, CVPR 2011), built from ImageNet.
**License:** ImageNet terms: non-commerical research and education only.
Images are not in this repo; download `images.tar` from the link above, extract it to `data/stanford_dogs/`, and
run `python -m dogbreeds.build_dataset`.

## Breeds and split

| Breed | WordNet ID | Train | Val | Test | Total |
|---|---|---|---|---|---|
| basset | n02088238 | 122 | 26 | 27 | 175 |
| bloodhound | n02088466 | 131 | 28 | 28 | 187 |
| border_collie | n02106166 | 105 | 22 | 23 | 150 |
| doberman | n02107142 | 105 | 22 | 23 | 150 |
| german_shepherd | n02106662 | 106 | 23 | 23 | 152 |
| golden_retriever | n02099601 | 105 | 22 | 23 | 150 |
| maltese | n02085936 | 176 | 38 | 38 | 252 |
| pug | n02110958 | 140 | 30 | 30 | 200 |
| rhodesian_ridgeback | n02087394 | 120 | 26 | 26 | 172 |
| shih_tzu | n02086240 | 150 | 32 | 32 | 214 |
| siberian_husky | n02110185 | 134 | 29 | 29 | 192 |
| yorkshire_terrier | n02094433 | 115 | 25 | 24 | 164 |
| **total** | | **1,509** | **323** | **326** | **2,158** |

- Stratified 70/15/15 split, seed 42, fixed in `data/splits.csv`.
Stanford's official split is not used.
- 3 duplicate pairs (2 Bloodhound, 1 Maltese) are kept, each pair inside one split.
- Breeds have 150-252 images (1.7x); the imbalance is handled in training.
- Same 12 breeds as the 2024 v1 project, including look-alike groups: Maltese / Shih-Tzu / Yorkshire Terrier , Basset / Bloodhound, Siberian Husky / German Shepherd.

## Known issues

- **Label noise:** a few likely wrong-breed images.
- **Busy photos:** people, toher dogs, dogs small or cut off.
- **Text and watermarks**: e.g. `n02107142_4663` (watermark), `n02107142_534` (sign reading "DOBERMAN").
- **Variety:** puppies and adults mixed; no mixed-breed dogs; 81 images under 224 px on the shortest side.
- **ImageNet overlap:** pretrained ImageNet modles have likely seen these exact photos, so test accuracy may be optimisitc

No images were removed; see `docs/decisions.md`

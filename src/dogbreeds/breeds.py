"""The 12 breeds in this project and their WordNet IDs in Stanford Dogs."""

BREED_WN_IDS = {
    "basset": "n02088238",
    "bloodhound": "n02088466",
    "border_collie": "n02106166",
    "doberman": "n02107142",
    "german_shepherd": "n02106662",
    "golden_retriever": "n02099601",
    "maltese": "n02085936",
    "pug": "n02110958",
    "rhodesian_ridgeback": "n02087394",
    "shih_tzu": "n02086240",
    "siberian_husky": "n02110185",
    "yorkshire_terrier": "n02094433",
}

# Class index = position in this list. Use it everywhere a label becomes a number.
CLASS_NAMES = sorted(BREED_WN_IDS)

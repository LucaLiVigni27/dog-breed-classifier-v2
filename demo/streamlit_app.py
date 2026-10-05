"""
Streamlit demo: upload a dog photo and get its breed (one of 12) or "not
sure".

Run locally from the repo root with:
    DOGBREEDS_CHECKPOINT=outputs/runs/convnext_tiny_v2/best.pt
    python -m streamlit run demo/streamlit_app.py
On Streamlit Community Cloud the weights are downloaded from the GitHub release instead.

"""

import os
import urllib.request
from pathlib import Path
from typing import cast

import streamlit as st
from PIL import Image, ImageOps
from streamlit_paste_button import paste_image_button

from dogbreeds.breeds import CLASS_NAMES, DISPLAY_NAMES
from dogbreeds.inference import NOT_SURE_THRESHOLD, load_classifier, predict_image

DEMO_DIR = Path(__file__).parent
EXAMPLES_DIR = DEMO_DIR / "examples"
# best.pt is too big for git, so the live app downloads it once from the GitHub release.
WEIGHTS_URL = (
    "https://github.com/LucaLiVigni27/dog-breed-classifier-v2/releases/download/v1.0/best.pt"
)
DOWNLOADED_WEIGHTS = DEMO_DIR / "best.pt"


def get_checkpoint_path() -> Path:
    """Use DOGBREEDS_CHECKPOINT if set (local, Docker); otherwise download the weights."""
    if "DOGBREEDS_CHECKPOINT" in os.environ:
        return Path(os.environ["DOGBREEDS_CHECKPOINT"])
    if not DOWNLOADED_WEIGHTS.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, DOWNLOADED_WEIGHTS)
    return DOWNLOADED_WEIGHTS


# Streamlit reruns this whole script on every click, so the model must be cached:
# it is loaded once per server instead of on every interaction.
@st.cache_resource
def get_classifier():
    return load_classifier(get_checkpoint_path())


def show_header() -> None:
    """Title plus a centred description, one sentence per line."""
    st.markdown("<h1 style='text-align: center'>Dog Breed Classifier</h1>", unsafe_allow_html=True)
    breeds = []
    for name in CLASS_NAMES:
        breeds.append(DISPLAY_NAMES[name])
    lines = [
        f"Recognises 12 breeds: {', '.join(breeds)}.",
        f'Below {NOT_SURE_THRESHOLD:.0%} confidence it answers "not sure".',
        "ConvNeXt-Tiny fine-tuned on Stanford Dogs; non-commercial, for education.",
    ]
    description = "<br>".join(lines)
    st.markdown(f"<div style='text-align: center'>{description}</div>", unsafe_allow_html=True)


def choose_example(path: Path) -> None:
    """Runs when an example's button is clicked, before the page reruns."""
    st.session_state["example"] = str(path)
    st.session_state["reset_count"] += 1


def show_examples() -> None:
    """One button per example photo; the chosen one is remembered between reruns."""
    paths = sorted(EXAMPLES_DIR.glob("*.jpg"))
    if not paths:
        return
    st.write("Or try an example:")
    columns = st.columns(len(paths))
    for column, path in zip(columns, paths, strict=True):
        with column:
            st.image(str(path))
            st.button("Try this", key=path.name, on_click=choose_example, args=(path,))


def show_result(result: dict) -> None:
    """Top 3 breeds as bars, then the verdict: the breed, or a large "Not sure"."""
    for _, display_name, probability in result["top"]:
        st.progress(probability, text=f"{display_name}: {probability:.0%}")
    name = result["display_name"]
    confidence = result["confidence"]
    if result["not_sure"]:
        st.header("Not sure")
        st.write(
            f"Best guess: {name} ({confidence:.0%}), but this is probably "
            "not one of the 12 breeds this model knows."
        )
    else:
        st.subheader(f"{name} ({confidence:.0%} confident)")


st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶")
show_header()
if "reset_count" not in st.session_state:
    st.session_state["reset_count"] = 0
reset_count = st.session_state["reset_count"]

uploaded = st.file_uploader(
    "Upload a dog photo", type=["jpg", "jpeg", "png", "webp"], key=f"upload_{reset_count}"
)
# Streamlit's uploader can't take pasted images, so this add-on button reads the clipboard.
pasted = paste_image_button("Paste an image from the clipboard", key=f"paste_{reset_count}")
show_examples()

# Priority when several are set: uploaded file, then pasted image, then clicked example.
image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif pasted.image_data is not None:
    image = cast(Image.Image, pasted.image_data)
elif "example" in st.session_state:
    image = Image.open(st.session_state["example"])
if image is not None:
    # Phone photos store rotation in EXIF; apply it so the photo is shown upright.
    image = ImageOps.exif_transpose(image)
    image_column, result_column = st.columns(2)
    with image_column:
        st.image(image)
    with result_column:
        show_result(predict_image(get_classifier(), image))

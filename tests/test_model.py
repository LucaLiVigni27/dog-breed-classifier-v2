import torch

from dogbreeds.model import build_model, get_head, set_backbone_trainable


def test_v1_baseline_output_shape():
    model = build_model("v1_baseline", 12, pretrained=False)
    logits = model(torch.zeros(2, 3, 200, 200))
    assert logits.shape == (2, 12)


def test_timm_model_output_shape():
    model = build_model("resnet18", 12, pretrained=False)
    logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 12)


def test_freeze_leaves_only_head_trainable():
    model = build_model("resnet18", 12, pretrained=False)
    head_params = set(get_head(model).parameters())

    set_backbone_trainable(model, False)
    for param in model.parameters():
        assert param.requires_grad == (param in head_params)

    set_backbone_trainable(model, True)
    for param in model.parameters():
        assert param.requires_grad

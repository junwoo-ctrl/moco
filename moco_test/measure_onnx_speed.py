
import random
import arrow
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

import onnxruntime
from clize import run
from PIL import Image

from moco_test.measure_inference_speed import get_labels


def fix_environment():

    seed = 9771
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    print("[  environment setting  ]")
    print("fix torch seeds with 9771. ")
    print("cudnn deteministic config True. \n")


def load_pretrained_model_cuda(is_gpu: bool):

    print("[  model load from pretrained weight file  ]")
    print("define model architect, freeze all layer, attach fc layers.")
    architecture_var: str = "resnet50"
    model = models.__dict__[architecture_var]()
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    print("load pretrained moco_v2 model weights.")
    print("delete renamed or unused layer name. \n")
    moco_v2_800_epoch_checkpoint_filename = "/opt/project/moco_test/pretrained_weights/moco_v2_800ep_pretrain.pth.tar"
    moco_v2_800_epoch_checkpoint = torch.load(moco_v2_800_epoch_checkpoint_filename)
    state_dict = moco_v2_800_epoch_checkpoint["state_dict"]
    for layer_name in list(state_dict.keys()):
        if layer_name.startswith("module.encoder_q") and not layer_name.startswith("module.encoder_q.fc"):
            state_dict[layer_name[len("module.encoder_q"):]] = state_dict[layer_name]
        del state_dict[layer_name]

    model.load_state_dict(state_dict, strict=False)
    if is_gpu:
        model = model.cuda()
    return model


def convert_torch_to_onnx(model: nn.Module, is_gpu: bool):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f="onnx_moco.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['modelInput'],
        output_names=['modelOutput'],
    )

    ort_session = onnxruntime.InferenceSession("onnx_moco.onnx")
    return ort_session


def get_sample_image_vector():
    sample_image_filepath: str = "/opt/project/moco_test/sample_image/sample_boat.jpeg"
    transform_config = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    sample_image_tensor: torch.Tensor = transform_config(Image.open(sample_image_filepath).convert("RGB"))
    sample_image_tensor = torch.unsqueeze(sample_image_tensor, dim=0)
    sample_image_vector = sample_image_tensor.cpu().numpy()
    return sample_image_vector


def execute_inference(ort: onnxruntime.InferenceSession, data: np.array):

    ort_inputs = {ort.get_inputs()[0].name: data}
    ort_outputs = ort.run(None, ort_inputs)
    return ort_outputs


def verbose_interval(start, end):
    interval = (end- start).seconds
    print(f"start time: {start}. ")
    print(f"end time: {end}. ")
    print(f"takes {interval} seconds.")
    return


def onnx_inference_cpu(*, verbose: bool = True):

    fix_environment()
    moco_v2: nn.Module = load_pretrained_model_cuda(is_gpu=False)
    ort_model_v2: Any = convert_torch_to_onnx(model=moco_v2, is_gpu=False)
    sample_image_vector: Any = get_sample_image_vector()

    print("[  inference for 10000 iterations  ]")
    start = arrow.utcnow()
    for i in range(0, 10000):
        execute_inference(ort=ort_model_v2, data=sample_image_vector)
    end = arrow.utcnow()
    verbose_interval(start=start, end=end)


if __name__ == "__main__":
    run({
        "onnx_inference": onnx_inference_cpu,
    })
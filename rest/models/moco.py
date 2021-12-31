
import random

import onnxruntime
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn

from PIL import Image


class MoCo:

    def __init__(self, is_gpu: bool = False):
        self.fix_environment()
        self.torch_model = self._load_torch_model(is_gpu=is_gpu)
        if not is_gpu:
            self.onnx_model = self._convert_torch_to_onnx()

    def fix_environment(self):
        seed = 9771
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

    def _convert_torch_to_onnx(self):
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

        torch.onnx.export(
            model=self.torch_model,
            args=dummy_input,
            f="rest_onnx_moco.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["modelInput"],
            output_names=["modelOutput"],
        )

        ort_seesion = onnxruntime.InferenceSession("rest_onnx_moco.onnx")
        return ort_seesion

    def _load_torch_model(self, is_gpu: bool = False):
        architecture_var: str = "resnet50"
        model = models.__dict__[architecture_var]()
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False

        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

        model_path = "/opt/project/moco_test/pretrained_weights/moco_v2_800ep_pretrain.pth.tar"
        model_checkpoint = torch.load(model_path)
        state_dict = model_checkpoint["state_dict"]

        for layer_name in list(state_dict.keys()):
            if layer_name.startswith("module.encoder_q") and not layer_name.startswith("module.encoder_q.fc"):
                state_dict[layer_name[len("module.encoder_q"):]] = state_dict[layer_name]
            del state_dict[layer_name]

        model.load_state_dict(state_dict, strict=False)
        if is_gpu:
            model = model.cuda()

        model.eval()
        return model

    def _convert_image_to_tensor(self, sample_image_path: str, is_gpu: bool = False):
        transform_config = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        sample_image_tensor: torch.Tensor = transform_config(Image.open(sample_image_path).convert("RGB"))
        sample_image_tensor = torch.unsqueeze(sample_image_tensor, dim=0)
        if is_gpu:
            sample_image_tensor = sample_image_tensor.cuda()
        return sample_image_tensor

    def inference(self, sample_image_path: str, is_gpu: bool = False):
        image_tensor = self._convert_image_to_tensor(sample_image_path=sample_image_path, is_gpu=is_gpu)
        hat = self.torch_model(image_tensor).to('cpu')
        _, preds = hat.topk(5, 1, True, True)
        preds = preds.t()
        preds = [int(pred[0]) for pred in preds]
        return preds

    def inference_onnx(self, sample_image_path: str):
        image_tensor = self._convert_image_to_tensor(sample_image_path=sample_image_path, is_gpu=False)
        image_vector = image_tensor.cpu().numpy()

        onnx_inputs = {self.onnx_model.get_inputs()[0].name: image_vector}
        onnx_outputs = self.onnx_model.run(None, onnx_inputs)[0]
        _, onnx_outputs = torch.Tensor(onnx_outputs).topk(5, 1, True, True)
        preds = onnx_outputs.t()
        preds = [int(pred[0]) for pred in preds]
        return preds
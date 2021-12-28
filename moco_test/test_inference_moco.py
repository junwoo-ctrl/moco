import random

import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torchinfo import summary


def inference(*, verbose: bool = True):

    # freeze seeds.
    seed = 9771
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


    # models
    backbone_arch: str = "resnet50"
    backbone = models.__dict__[backbone_arch]()
    print("[  verbose model side  ]")
    print(f"create model {backbone_arch}.")

    print("freeze all layers but attach the last fc.")
    for name, param in backbone.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    print("init the fc layer.")
    backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
    backbone.fc.bias.data.zero_()
    print("\n")

    # model loads
    print("[  load model from checkpoint file  ]")
    model_checkpoint_filepath = "/opt/project/moco_test/pretrained_weights/moco_v2_800ep_pretrain.pth.tar"
    checkpoint = torch.load(model_checkpoint_filepath)
    print("", checkpoint["state_dict"].keys())
    print(checkpoint["arch"])

    # retain only encoder_q up to before the embedding layer
    state_dict = checkpoint["state_dict"]
    for layer_name in list(state_dict.keys()):
        if layer_name.startswith("module.encoder_q") and not layer_name.startswith("module.encoder_q.fc"):
            state_dict[layer_name[len("module.encoder_q."):]] = state_dict[layer_name]

        # delete renamed or unused layer name
        del state_dict[layer_name]

    backbone.load_state_dict(checkpoint["state_dict"], strict=False)
    print("loading model checkpoint.")
    print("model sumary.")
    print(summary(backbone))
    print('is_cuda: ', torch.cuda.is_available())
    print("\n")

    # gpu settings (1 gpu  with single machine.
    print("[  gpu settings  ]")
    gpus = torch.cuda.device_count()
    model = backbone.cuda()
    print(f"num of available gpus: {gpus}")
    print("successed load model on gpu.")
    print("\n\n")
    return None


if __name__ == "__main__":
    inference()
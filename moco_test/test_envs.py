
from clize import run


def test_env(*, verbose: bool = True):

    print("[  pytorch available, cuda setting  ]")
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    import torch.optim
    import torch.multiprocessing as mp
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torchvision.models as models

    print("is cuda avaliable: ", torch.cuda.is_available())


    print("[  set PYTHONPATH for moco  ]")
    import moco.loader
    import moco.builder

    print("moco loader: ", moco.loader)
    print("moco builder: ", moco.builder)


if __name__ == "__main__":
    run({
        'test_env': test_env,
    })
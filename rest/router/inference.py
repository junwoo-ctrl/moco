
from fastapi import APIRouter

from rest.models.moco import MoCo


router = APIRouter()
moco_cpu = MoCo(is_gpu=False)
moco_gpu = MoCo(is_gpu=True)


@router.get("/inference" + "/moco/torch-cpu", tags=["torch", "moco", "cpu"])
def execute_inference_moco_torch_cpu(sample_image_path: str):
    inference_result = moco_cpu.inference(sample_image_path=sample_image_path, is_gpu=False)
    return {
        "type": "torch-cpu",
        "inference_result": inference_result,
    }


@router.get("/inference" + "/moco/torch-gpu", tags=["torch", "moco", "gpu"])
def execute_inference_moco_torch_gpu(sample_image_path: str):
    inference_result = moco_gpu.inference(sample_image_path=sample_image_path, is_gpu=True)
    return {
        "type": "torch-gpu",
        "inference_result": inference_result,
    }


@router.get("/inference" + "/moco/onnx", tags=["onnx", "moco", "cpu"])
def execute_inference_moco_onnx(sample_image_path: str):
    inference_result = moco_cpu.inference_onnx(sample_image_path=sample_image_path)
    return {
        "type": "onnx",
        "inference_result": inference_result,
    }
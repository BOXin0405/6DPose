# export_hccepose_onnx.py
import os
import torch
from HccePose.network_model import HccePose_BF_Net, load_checkpoint

def export_hccepose_onnx(
    checkpoint_path: str = "demo-bin-picking/HccePose/obj_01/best_score",
    onnx_output_path: str = "./onnx_models/HccePose_BF_Net.onnx",
    input_size=(3, 256, 256),
    batch_size=1,
    efficientnet_key=None,
    cuda_device='0'
):
    # 设备
    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = HccePose_BF_Net(efficientnet_key=efficientnet_key).to(device)
    model.eval()
    
    # 加载 checkpoint
    checkpoint_info = load_checkpoint(checkpoint_path, model, CUDA_DEVICE=cuda_device)
    print(f"Loaded checkpoint: best_score={checkpoint_info['best_score']}, iteration_step={checkpoint_info['iteration_step']}")

    # 创建虚拟输入
    dummy_input = torch.randn(batch_size, *input_size, device=device)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=['input'],
        output_names=['mask', 'binary_code'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'mask': {0: 'batch_size'},
            'binary_code': {0: 'batch_size'}
        },
        opset_version=16
    )
    print(f"ONNX model exported to {onnx_output_path}")
    print(f"Input shape: {dummy_input.shape}")
    print("Output shapes: mask (B,1,128,128), binary_code (B,48,128,128)")

if __name__ == "__main__":
    export_hccepose_onnx()

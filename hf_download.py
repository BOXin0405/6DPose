import os
import shutil
from huggingface_hub import hf_hub_download

# 仓库 ID
REPO_ID = "SEU-WYL/HccePose"

# 要下载的文件列表（保持原仓库相对路径）
FILES_TO_DOWNLOAD = [
    "demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500",
    "demo-bin-picking/models/models_info.json",
    "demo-bin-picking/models/obj_000001.ply",
    "demo-bin-picking/models/obj_000001_sym.ply",
    "demo-bin-picking/models/obj_000001_sym_type.json",
    "demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt",
]

# 本地保存根目录
SAVE_ROOT = "./"

# 单独缓存目录，不放在目标文件夹中
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

for f in FILES_TO_DOWNLOAD:
    # 构建本地保存完整路径
    local_path = os.path.join(SAVE_ROOT, f)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 下载文件到单独缓存目录
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f,
        repo_type="dataset",
        cache_dir=CACHE_DIR,
        resume_download=True
    )

    # 复制到目标目录
    shutil.copy(downloaded_path, local_path)
    print(f"Downloaded and saved: {local_path}")

print("\nAll files downloaded successfully!")

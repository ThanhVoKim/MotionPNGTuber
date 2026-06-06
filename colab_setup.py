# MotionPNGTuber — Google Colab CPU Setup
#
# Cách dùng: Copy-paste TỪNG ĐOẠN (từ `# %%` này đến `# %%` tiếp theo)
# vào một code cell riêng trong Colab rồi chạy theo thứ tự từ 1 đến 10.
#
# Lưu ý quan trọng:
#   - Nếu runtime bị restart giữa chừng: chạy lại cell 2 (PATH) trước khi tiếp tục.
#   - Cell 6 (build mmcv-full) mất ~5-10 phút, đừng ngắt.
#   - Lần đầu chạy detector (cell 9), anime-face-detector tải thêm weights ~300 MB.
#   - Script này KHÔNG cần GPU. Detector chạy bằng CPU.
#   - Mọi "uv pip install" đều truyền "--python .venv/bin/python": trong Colab venv
#     không được activate, nếu không ép thì uv cài nhầm vào Python 3.12 hệ thống
#     (torch 1.13 không có wheel cp312 → lỗi). Phải chạy cell 4 trước các cell cài.

# %% [markdown]
# ## MotionPNGTuber — Colab CPU Setup
# Copy-paste từng cell `# %%` vào một code cell riêng rồi chạy theo thứ tự.

# %% 1 — Kiểm tra môi trường
import subprocess, sys

print("=== Python version ===")
print(sys.version)

print("\n=== GPU info (không bắt buộc) ===")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines()[:6]:
            print(line)
    else:
        print("No GPU detected — OK, detector sẽ chạy bằng CPU.")
except FileNotFoundError:
    print("nvidia-smi không có — OK, CPU-only runtime, detector sẽ chạy bằng CPU.")

print("\n=== CUDA (torch check sẽ chạy ở cell 8) ===")
print("Môi trường OK.")

# %% 2 — Cài uv và thêm vào PATH (chạy lại sau khi restart runtime)
import os, subprocess

print("Đang cài uv...")
subprocess.run(
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    shell=True, check=True
)

# Thêm uv vào PATH của kernel (tồn tại trong suốt session Colab hiện tại)
uv_bin = "/root/.local/bin"
if uv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = uv_bin + ":" + os.environ.get("PATH", "")

# Xác nhận
result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
print("uv version:", result.stdout.strip())

# %% 3 — Clone repo (bỏ qua nếu đã có)
import os, subprocess

REPO_URL = "https://github.com/ThanhVoKim/MotionPNGTuber.git"
REPO_DIR = "/content/MotionPNGTuber"

if os.path.isdir(REPO_DIR):
    print(f"Repo đã có tại {REPO_DIR}, bỏ qua bước clone.")
else:
    print("Đang clone repo...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("Clone xong.")

# Chuyển vào thư mục repo (tồn tại cho các cell tiếp theo trong session)
os.chdir(REPO_DIR)
print(f"Thư mục hiện tại: {os.getcwd()}")

# %% 4 — Tạo venv Python 3.10 + build dependencies
# uv tự tải Python 3.10 nếu chưa có (~60 MB, chỉ 1 lần).
# setuptools<81 là bắt buộc: setuptools>=82 đã xóa pkg_resources mà mmcv-full cần để build.
#
# QUAN TRỌNG — vì sao mọi "uv pip install" đều phải có "--python .venv/bin/python":
#   Trong Colab venv KHÔNG được activate, nên mặc định uv cài vào Python 3.12
#   hệ thống (/usr) → torch 1.13 không có wheel cp312 → lỗi.
#   Set VIRTUAL_ENV + truyền "--python .venv/bin/python" ép uv dùng đúng venv 3.10.
import os, subprocess

print("Tạo venv Python 3.10...")
subprocess.run(["uv", "venv", "--python", "3.10"], check=True)

# Trỏ uv vào venv vừa tạo (tồn tại cho các cell sau trong session).
VENV_PY = os.path.abspath(".venv/bin/python")
os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")

print("\nCài build dependencies...")
subprocess.run([
    "uv", "pip", "install", "--python", VENV_PY,
    "setuptools<81", "wheel", "pip", "ninja", "cython",
    "numpy>=1.24,<2.0",
], check=True)

# Xác nhận Python trong venv
result = subprocess.run(
    [VENV_PY, "--version"], capture_output=True, text=True
)
print("Python trong venv:", result.stdout.strip())

# %% 5 — Cài PyTorch 1.13.1+cpu
# Dùng bản +cpu (không cần CUDA): nhẹ hơn 10x (~200 MB vs 2.4 GB).
# mmcv-full 1.7.0 tương thích với torch 1.13.x bất kể +cpu hay +cu117.
#
# QUAN TRỌNG — vì sao cần CẢ HAI index:
#   - torch/torchvision bản "+cpu" CHỈ có trên index pytorch.
#   - Nhưng torchvision còn phụ thuộc pillow/requests/typing-extensions,
#     những thứ này KHÔNG có trên index pytorch → phải lấy từ PyPI.
#   - "--index-url" thay thế PyPI hoàn toàn, nên phải thêm "--extra-index-url"
#     trỏ về PyPI + "--index-strategy unsafe-best-match" để uv tra cả hai.
#   - Ghim cứng "+cpu" nên không sợ lấy nhầm torch từ PyPI (PyPI không có +cpu).
# "--python .venv/bin/python": ép cài vào venv 3.10 (xem ghi chú cell 4).
import os, subprocess

VENV_PY = os.path.abspath(".venv/bin/python")
os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")

print("Đang cài torch 1.13.1+cpu và torchvision... (~200 MB)")
result = subprocess.run([
    "uv", "pip", "install", "--python", VENV_PY,
    "torch==1.13.1+cpu",
    "torchvision==0.14.1+cpu",
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--extra-index-url", "https://pypi.org/simple",
    "--index-strategy", "unsafe-best-match",
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("=== LỖI khi cài torch (stderr) ===")
    print(result.stderr)
    raise SystemExit("torch install thất bại — xem stderr ở trên.")
print("torch cài xong.")

# %% 6 — Cài mmcv-full 1.7.0 từ pre-built wheel (~25 MB, ~30 giây)
import os, subprocess

VENV_PY = os.path.abspath(".venv/bin/python")
os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")

WHEEL_URL = (
    "https://github.com/ThanhVoKim/mmcv-wheels/releases/download/"
    "wheels-mmcv1.7.0/mmcv_full-1.7.0-cp310-cp310-linux_x86_64.whl"
)

print("Cài mmcv-full 1.7.0 từ pre-built wheel (~25 MB)...")
result = subprocess.run([
    "uv", "pip", "install", "--python", VENV_PY, WHEEL_URL,
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("=== LỖI (stderr) ===")
    print(result.stderr)
    raise SystemExit("mmcv-full install thất bại.")
print("mmcv-full cài xong.")

# %% 6b — (ĐÃ DÙNG / KHÔNG CẦN CHẠY) Xuất wheel để upload GitHub Releases
# Wheel Linux đã được host tại ThanhVoKim/mmcv-wheels — cell 6 cài trực tiếp từ đó.
# Chỉ chạy cell này nếu cần rebuild wheel (ví dụ: runtime Colab thay đổi Python/ABI).
import os, subprocess

VENV_PY = os.path.abspath(".venv/bin/python")
os.environ["MMCV_WITH_OPS"] = "1"
os.environ["FORCE_CUDA"] = "0"

os.makedirs("/content/wheels", exist_ok=True)
print("Đang build wheel để export...")
subprocess.run([
    VENV_PY, "-m", "pip", "wheel",
    "--no-build-isolation",
    "--no-deps",
    "-w", "/content/wheels",
    "mmcv-full==1.7.0",
], check=True)

import glob
wheels = glob.glob("/content/wheels/*.whl")
print("Wheel xuất ra:")
for w in wheels:
    size_mb = os.path.getsize(w) / 1e6
    print(f"  {w}  ({size_mb:.1f} MB)")

print("\nTải file này về máy rồi upload lên GitHub Releases.")
print("Sau đó cập nhật URL trong pyproject.toml > [tool.uv.sources] > mmcv-full.")

# Tải về máy (bỏ comment dòng dưới nếu muốn auto-download)
# from google.colab import files; files.download(wheels[0])

# %% 7 — Cài mmdet, mmpose, anime-face-detector và các deps
# --python .venv/bin/python : cài vào venv 3.10 (xem ghi chú cell 4)
import os, subprocess

VENV_PY = os.path.abspath(".venv/bin/python")
os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")

print("Đang cài mmdet 2.28.0 và mmpose 0.29.0...")
subprocess.run([
    "uv", "pip", "install", "--python", VENV_PY,
    "--no-build-isolation",   # cần cho chumpy (dep của mmpose) — cùng lý do setuptools
    "mmdet==2.28.0",
    "mmpose==0.29.0",
], check=True)

print("\nĐang cài anime-face-detector và dependencies còn lại...")
subprocess.run([
    "uv", "pip", "install", "--python", VENV_PY,
    "anime-face-detector==0.0.9",
    "opencv-python",
    "scipy",
    "sounddevice",
    "pillow",
    "openmim",
    "addict",
    "yapf",
    "tqdm",
], check=True)
print("Tất cả dependencies đã cài xong.")

# %% 8 — Kiểm tra import
# MPLBACKEND=Agg: Colab set MPLBACKEND=module://matplotlib_inline.backend_inline,
#   nhưng backend đó (gói matplotlib-inline) chỉ có ở Python hệ thống, KHÔNG có
#   trong venv 3.10. mmdet import matplotlib → lỗi "not a valid backend".
#   Ép Agg (backend không cần GUI, luôn có sẵn) cho subprocess venv.
import os, subprocess, sys

venv_env = {**os.environ, "MPLBACKEND": "Agg"}

check_code = """
import torch, mmcv, mmdet, mmpose
from anime_face_detector import create_detector

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
print("anime_face_detector: OK")
print()
print("=== Tất cả import thành công! ===")
"""

result = subprocess.run(
    [".venv/bin/python", "-c", check_code],
    capture_output=True, text=True, env=venv_env,
)
print(result.stdout)
if result.returncode != 0:
    print("LỖI:", result.stderr[-2000:])

# %% 9 — (TÙY CHỌN) Upload video và chạy face detector
# Tải lên file video .mp4 của bạn rồi cell này sẽ chạy detector và xuất mouth_track.npz
# MPLBACKEND=Agg: xem ghi chú cell 8 (detector cũng import mmdet → matplotlib).
from google.colab import files
import subprocess, os

venv_env = {**os.environ, "MPLBACKEND": "Agg"}

# Tự dò các flag mà bản face_track_anime_detector.py hiện tại hỗ trợ.
# (Bản trên GitHub có thể khác bản local, nên chỉ truyền flag thực sự có.)
help_txt = subprocess.run(
    [".venv/bin/python", "face_track_anime_detector.py", "--help"],
    capture_output=True, text=True, env=venv_env,
).stdout

print("Chọn file video .mp4 để upload...")
uploaded = files.upload()

if not uploaded:
    print("Không có file nào được upload.")
else:
    video_path = list(uploaded.keys())[0]
    out_path = "mouth_track.npz"
    debug_path = "mouth_track_debug.mp4"

    cmd = [
        ".venv/bin/python", "face_track_anime_detector.py",
        "--video", video_path,
        "--out", out_path,
        "--device", "cpu",       # CPU mode
    ]
    # Chỉ thêm flag tăng tốc / debug nếu script hỗ trợ
    if "--debug" in help_txt:
        cmd += ["--debug", debug_path]
    if "--stride" in help_txt:
        cmd += ["--stride", "2"]        # xử lý mỗi 2 frame cho nhanh
    if "--det-scale" in help_txt:
        cmd += ["--det-scale", "0.75"]  # scale nhỏ hơn cho nhanh

    print(f"\nVideo: {video_path}")
    print("Lệnh:", " ".join(cmd))
    print("Đang chạy face detector (CPU)...")
    print("Lần đầu chạy: anime-face-detector sẽ tải thêm model weights (~300 MB).\n")

    # Bắt output để nếu lỗi còn thấy lý do thật (Colab hay nuốt stderr khi không bắt).
    result = subprocess.run(cmd, capture_output=True, text=True, env=venv_env)
    print(result.stdout)
    if result.returncode == 0:
        size = os.path.getsize(out_path)
        print(f"\nXong! {out_path} ({size:,} bytes)")
    else:
        print(f"\n=== LỖI: returncode={result.returncode} (stderr) ===")
        print(result.stderr[-4000:])

# %% 10 — (TÙY CHỌN) Tải kết quả về máy
import os
from google.colab import files

for fname in ["mouth_track.npz", "mouth_track_debug.mp4"]:
    if os.path.isfile(fname):
        size_mb = os.path.getsize(fname) / 1e6
        print(f"Tải {fname} ({size_mb:.1f} MB)...")
        files.download(fname)
    else:
        print(f"{fname}: không tìm thấy (bỏ qua).")

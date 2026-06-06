# Khôi phục cài đặt MotionPNGTuber (domain openmmlab đã sập) + hỗ trợ Colab/laptop CPU

## Bối cảnh (tại sao cần làm)

- `download.openmmlab.com` đã **sập trên phạm vi toàn cầu** (fetch từ ngoài Colab cũng `ECONNREFUSED`). [pyproject.toml](../pyproject.toml#L47-L50) và [uv.lock](../uv.lock) ghim cứng 2 URL wheel `mmcv-full==1.7.0` vào domain này ⇒ **mọi `uv sync` mới đều hỏng trên cả Windows lẫn Linux**, không riêng Colab.
- PyPI chỉ có **sdist** cho `mmcv-full 1.7.0` ⇒ phải build từ source. Build hỏng vì: (1) `setuptools>=82` đã bỏ `pkg_resources` ⇒ cần `setuptools<81` + `--no-build-isolation`; (2) môi trường Colab mặc định lệch (Python 3.12 / torch 2.x / numpy 2) trong khi `mmcv-full 1.7.0` chỉ build được với **Python 3.10 + torch 1.13**.
- Khả năng cứu wheel: **wheel Windows cu117 còn trên Internet Archive** (snapshot `20260113022446`); **wheel Linux KHÔNG còn** (Wayback trống cả `cu117` lẫn `cpu`).
- Site `miropsota.github.io/torch_packages_builder/mmcv`: chỉ có **mmcv 2.2.0 cho torch 2.x** (API mmcv 2.x) ⇒ **KHÔNG tương thích** stack khóa cứng của dự án (`anime-face-detector 0.0.9` + `mmdet 2.28` + `mmpose 0.29` đều dùng API mmcv 1.x, yêu cầu `mmcv-full` trong `[1.3.17, 1.8.0)`). **Không dùng được.**
- Đã chốt với người dùng: (1) **Colab + sửa repo**; (2) Colab chạy **CPU**; (3) lâu dài **re-host wheel lên GitHub Releases**.
- **Laptop Python 3.13.2 KHÔNG phải vấn đề**: dự án khóa `requires-python = "==3.10.*"`, `uv` **tự tải Python 3.10**. ⇒ cần wheel **cp310**, không phải cp313. Laptop CPU vẫn cài torch `1.13.1+cu117` (chạy được trên CPU, ~2.4GB) và dùng wheel `win_amd64` cu117 đã cứu được.

## Kết quả bàn giao

### A. `colab_setup.py` (MỚI) — file Python chia cell `# %%` để copy-paste từng phần lên Colab

Cài CPU + build `mmcv-full` từ source ngay trong session (chạy được **ngay hôm nay**, không cần chờ hosting). Cell 6b tùy chọn xuất ra wheel `.whl` để đem đi host (Phần B/D). Cấu trúc cell:

```python
# %% [markdown]
# MotionPNGTuber — Colab CPU setup. Copy-paste TỪNG cell `# %%` vào Colab.

# %% 1. Môi trường
!python --version
!nvidia-smi || echo "no gpu -> ok, chay CPU"

# %% 2. Cài uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os; os.environ["PATH"] = "/root/.local/bin:" + os.environ["PATH"]
!uv --version

# %% 3. Clone repo
!git clone https://github.com/rotejin/MotionPNGTuber.git
%cd /content/MotionPNGTuber

# %% 4. venv Python 3.10 + build-deps (uv tự dùng .venv trong thư mục này)
!uv venv --python 3.10
!uv pip install "setuptools<81" wheel pip ninja cython "numpy>=1.24,<2.0"

# %% 5. torch 1.13.1+cu117 (khớp dự án; detector vẫn chạy CPU)
!uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# %% 6. Build mmcv-full 1.7.0 (CPU ops) — ~5-10 phút
!MMCV_WITH_OPS=1 FORCE_CUDA=0 uv pip install --no-build-isolation mmcv-full==1.7.0

# %% 6b. (TÙY CHỌN) Xuất wheel để upload GitHub Releases
!MMCV_WITH_OPS=1 FORCE_CUDA=0 .venv/bin/python -m pip wheel \
    --no-build-isolation --no-deps -w /content/wheels mmcv-full==1.7.0
!ls -la /content/wheels   # tải file .whl này về máy để host

# %% 7. mmdet / mmpose / anime-face-detector + deps nhẹ
!uv pip install --no-build-isolation mmdet==2.28.0 mmpose==0.29.0
!uv pip install anime-face-detector==0.0.9 opencv-python scipy sounddevice pillow openmim addict yapf

# %% 8. Kiểm tra import
!.venv/bin/python -c "import torch,mmcv,mmdet,mmpose,anime_face_detector as a; print('OK', torch.__version__, mmcv.__version__)"

# %% 9. Chạy detector (CPU)
from google.colab import files; up = files.upload()          # chọn video .mp4
VID = list(up.keys())[0]
!.venv/bin/python face_track_anime_detector.py --video "$VID" --out mouth_track.npz --device cpu --debug debug.mp4

# %% 10. Tải kết quả
from google.colab import files; files.download('mouth_track.npz')
```

Ghi chú trong file: `uv venv` không có sẵn `pip` ⇒ cell 6b cần `.venv/bin/python -m pip` (đã cài `pip` ở cell 4). Lần chạy detector đầu tiên `anime-face-detector` sẽ tải weights (yolov3/hrnet) về — cần mạng. `chumpy` (do mmpose kéo theo) chỉ dùng cho 3D mesh, không ảnh hưởng đường đi 2D của detector.

### B. Sửa repo cho `uv sync` chạy lại được mọi nơi

[pyproject.toml](../pyproject.toml#L47-L50) — đổi 2 URL `mmcv-full` trong `[tool.uv.sources]` từ `download.openmmlab.com` sang **GitHub Releases của repo** (sau khi host ở Phần D):

```toml
mmcv-full = [
    { url = "https://github.com/<OWNER>/MotionPNGTuber/releases/download/wheels-mmcv1.7.0/mmcv_full-1.7.0-cp310-cp310-linux_x86_64.whl", marker = "sys_platform == 'linux'" },
    { url = "https://github.com/<OWNER>/MotionPNGTuber/releases/download/wheels-mmcv1.7.0/mmcv_full-1.7.0-cp310-cp310-win_amd64.whl",   marker = "sys_platform == 'win32'" },
]
```

- **Stopgap Windows tức thì (nếu chưa kịp host):** tạm trỏ URL `win32` vào bản Wayback raw (`id_` để lấy đúng bytes file, KHÔNG ra trang openmmlab):
  `https://web.archive.org/web/20260113022446id_/https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.0-cp310-cp310-win_amd64.whl`
- Sau khi sửa: chạy `uv lock` để cập nhật [uv.lock](../uv.lock). Giữ nguyên `torch==1.13.1+cu117`, `requires-python="==3.10.*"`, index pytorch-cu117 (đều còn sống).
- Kiểm tra [pyproject.macos.toml](../pyproject.macos.toml) xem có tham chiếu URL chết không (macOS vốn build từ source theo README.ja ⇒ thường không cần đổi).

### C. Cập nhật README ([README.md](../README.md), [README.ja.md](../README.ja.md))

- Thêm chú ý nổi bật: domain openmmlab offline; wheel `mmcv-full` nay host ở **Releases của repo**.
- Mục cài Windows/Ubuntu: lệnh `uv sync` giữ nguyên (nay resolve từ Releases).
- Thêm mục **Google Colab (CPU)** trỏ tới `colab_setup.py`.
- Thêm chú ý **laptop**: Python 3.13 vẫn được (uv tự cấp 3.10); torch cu117 ~2.4GB; detector chạy CPU (đã có sẵn fallback CUDA→CPU trong [face_track_anime_detector.py](../face_track_anime_detector.py#L83-L117)).

### D. Các bước thủ công (maintainer) để tạo & host wheel — kèm lệnh

1. **Wheel Windows** (cp310, cu117): tải từ Wayback raw rồi kiểm tra:
   ```bash
   curl -L -o mmcv_full-1.7.0-cp310-cp310-win_amd64.whl \
     "https://web.archive.org/web/20260113022446id_/https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.0-cp310-cp310-win_amd64.whl"
   python -c "import zipfile; print(zipfile.ZipFile('mmcv_full-1.7.0-cp310-cp310-win_amd64.whl').namelist()[:3])"
   ```
   _Fallback nếu Wayback lỗi:_ lấy từ cache uv của lần cài Windows trước (`%LOCALAPPDATA%\uv\cache`).
2. **Wheel Linux** (cp310): lấy `/content/wheels/mmcv_full-1.7.0-cp310-cp310-linux_x86_64.whl` từ cell 6b ở Colab về máy.
3. **Tạo release & upload** (trên repo bạn có quyền publish):
   ```bash
   gh release create wheels-mmcv1.7.0 \
     mmcv_full-1.7.0-cp310-cp310-win_amd64.whl \
     mmcv_full-1.7.0-cp310-cp310-linux_x86_64.whl \
     -t "mmcv-full 1.7.0 wheels" -n "Re-host (openmmlab domain offline)"
   ```
4. Báo lại `<OWNER>` + tag để mình điền URL chính xác vào pyproject rồi `uv lock`.

## Files thay đổi / tạo mới

- **Sửa:** [pyproject.toml](../pyproject.toml) (2 URL `mmcv-full`), [uv.lock](../uv.lock) (qua `uv lock`), [README.md](../README.md), [README.ja.md](../README.ja.md). Kiểm tra [pyproject.macos.toml](../pyproject.macos.toml).
- **Tạo:** `colab_setup.py` (cell `# %%` cho Colab).

## Kiểm thử (verification)

- **Laptop (Windows, Py3.13, CPU):** `uv sync` → `uv run python -c "import anime_face_detector, mmcv, torch; print('OK')"` → `uv run face-track-detect --video sample.mp4 --out out.npz --device auto` (auto sẽ fallback về CPU).
- **Colab:** chạy `colab_setup.py` từ cell 1→8 (import OK), cell 9 chạy detector trên 1 video ngắn ra `mouth_track.npz`.
- **Repo:** sau khi host wheel + `uv lock`, `uv sync` sạch trên cả 2 nền tảng không còn gọi `download.openmmlab.com`.

## Cần bạn xác nhận khi triển khai

- **Repo nào sẽ host Release?** origin hiện là `rotejin/MotionPNGTuber`; bạn (ThanhVoKim) publish release ở đó được không, hay dùng fork `ThanhVoKim/MotionPNGTuber`? `<OWNER>` trong URL sẽ điền theo lựa chọn này.

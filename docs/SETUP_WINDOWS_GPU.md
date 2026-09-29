# Cài FaceSlim trên máy Windows có GPU NVIDIA (RTX 3080)

Tài liệu này dành cho một máy mới: clone source, chạy một lệnh setup, rồi mở app với đầy đủ flow chính (warp khuôn mặt, AI Beauty, webcam, export) cộng với flow tách răng bằng SAM 2 chạy trên GPU.

## 1. Chuẩn bị (một lần)

| Cần | Ghi chú |
|---|---|
| Windows 10/11 64-bit | |
| Driver NVIDIA mới | Cài từ [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx). Không cần cài CUDA Toolkit, PyTorch tự mang CUDA runtime. |
| Python 3.11 | Tải từ [python.org](https://www.python.org/downloads/release/python-3119/). Khi cài, tick **Add python.exe to PATH** và **py launcher**. Không dùng Python 3.13+ vì các thư viện đã pin (NumPy 1.26, ONNX Runtime 1.18) chưa có bản cho 3.13. |
| Git | [git-scm.com](https://git-scm.com/download/win) |
| Dung lượng trống | Khoảng 8 GB (PyTorch CUDA khoảng 3 GB, model khoảng 220 MB) |
| Internet | Chỉ cần lúc setup; sau đó app chạy offline |
| Tùy chọn | [FFmpeg](https://ffmpeg.org/) trong `PATH` để video export giữ âm thanh; OBS Virtual Camera nếu muốn xuất ra camera ảo |

Kiểm tra nhanh trong PowerShell:

```powershell
py -3.11 --version
git --version
nvidia-smi
```

## 2. Clone và setup

```powershell
git clone <url-repo-cua-ban> FaceSlim
cd FaceSlim
.\setup.bat
```

`setup.bat` tìm Python 3.11 (hoặc 3.12/3.10) rồi chạy `tools\setup_machine.py`. Script này:

1. Tạo `.venv` trong thư mục repo và cài `requirements.txt`.
2. Cài PyTorch bản CUDA. Script thử lần lượt `cu128`, `cu126`, `cu130` và chỉ giữ bản chạy được kernel thật trên GPU của máy. RTX 3080 thường nhận ngay `cu128`.
3. Clone [SAM 2](https://github.com/facebookresearch/sam2) vào `third_party\sam2` (commit cố định) và cài vào `.venv`, bỏ qua CUDA extension tùy chọn (FaceSlim không cần).
4. Tải trước và kiểm tra SHA-256 các model: MediaPipe Face Landmarker, BiSeNet ResNet18, SAM 2.1 Hiera Tiny.
5. Chạy `tools\doctor.py` để kiểm tra toàn bộ và đo tốc độ SAM 2.

Kết quả mong đợi ở cuối:

```text
  [OK  ] CUDA GPU: NVIDIA GeForce RTX 3080, compute 8.6, SAM 2 precision bf16
  [OK  ] Model SAM 2.1 Hiera Tiny: verified
  [OK  ] SAM 2 benchmark: ... ms per mouth crop on cuda (...)
Ready.
```

Setup chạy lại bao nhiêu lần cũng được. Bước nào đã xong sẽ được giữ nguyên, ví dụ PyTorch CUDA đang chạy tốt thì không cài lại.

Tùy chọn:

```powershell
.\setup.bat --cuda cu126     # ép một bản CUDA cụ thể
.\setup.bat --cuda cpu       # máy không có GPU NVIDIA
.\setup.bat --skip-sam2      # chỉ flow chính, không SAM 2
```

## 3. Chạy app

```powershell
.\run.bat
```

Hoặc `.venv\Scripts\python FaceSlim_v1.py`. Góc phải thanh trạng thái hiển thị `GPU: NVIDIA GeForce RTX 3080` khi warp đang chạy trên GPU (`CPU Mode` nếu không).

### Tẩy trắng răng bằng SAM 2

1. Bấm **Webcam**, hoặc **Load File** để mở ảnh/video.
2. Trong nhóm **AI Beauty**, kéo **Teeth Whitening** lên (ví dụ 60).
3. Ở dropdown **Teeth Mask:** ngay dưới, chọn **SAM 2 (landmark prompts)**.
4. Tùy chọn: trong nhóm **Quality**, tick checkbox **Teeth Mask** để phủ màu lên vùng sẽ được làm trắng, dễ kiểm tra.

Hai nguồn mask răng:

| Teeth Mask | Cách hoạt động |
|---|---|
| Face parsing (BiSeNet), mặc định | Model BiSeNet gán nhãn "mouth" cho toàn bộ phần trong miệng, gồm cả khóe tối, lưỡi, lợi. |
| SAM 2 (landmark prompts) | MediaPipe cho 478 landmark. Từ viền môi trong, FaceSlim tạo prompt cho SAM 2: một box quanh khe miệng, điểm dương trên răng trên (có lọc theo độ sáng), điểm âm trên môi, khóe miệng, nhân trung, cằm. SAM 2 tách đúng phần răng, sau đó mask được cắt theo viền môi trong. Miệng đóng thì bỏ qua SAM. |

Nếu SAM 2 không load được (chưa cài, lỗi GPU), app tự quay về mask BiSeNet và in lý do ra console.

### Realtime (webcam) và export

- **Webcam / xem trước video**: SAM 2 chạy trên một thread nền. Khung hình không bao giờ chờ SAM; mask mới nhất được dịch theo chuyển động miệng cho tới khi có kết quả kế tiếp. Vài frame đầu, lúc model đang load, dùng mask BiSeNet. Trên RTX 3080, SAM chạy bf16; trên máy thử GTX 1080 Ti (fp32), mỗi lần SAM mất khoảng 110 ms và chỉ làm thời gian mỗi frame tăng khoảng 5%.
- **Export video / batch / CLI**: chạy đồng bộ để kết quả mỗi frame cố định: SAM chạy mỗi 4 frame, giữa các lần đó mask dịch theo miệng.
- Một model SAM 2 được dùng chung cho cả tiến trình, nên bật/tắt webcam hoặc tua video không load lại model.

### Dòng lệnh

```powershell
# Ảnh
.venv\Scripts\python FaceSlim_v1.py -i smile.jpg -o out --preset Beauty --teeth-whiten 60 --teeth-segmenter sam2

# Video
.venv\Scripts\python FaceSlim_v1.py -i clip.mp4 -o out --teeth-whiten 60 --teeth-segmenter sam2

# Xem prompt và mask SAM 2 trên một ảnh (xuất ra <ten_anh>_sam2_teeth.png)
.venv\Scripts\python tools\sam2_teeth_debug.py smile.jpg
```

Trong ảnh debug: chấm xanh lá là điểm dương (răng), chấm đỏ là điểm âm, khung cam là box, vùng xanh dương là mask.

## 4. Cập nhật source

```powershell
git pull
.\setup.bat
```

## 5. Xử lý sự cố

Chạy `.venv\Scripts\python tools\doctor.py` trước; mỗi dòng `FAIL`/`WARN` chỉ ra phần hỏng.

| Triệu chứng | Cách xử lý |
|---|---|
| `Python 3.10-3.12 was not found` | Cài Python 3.11 kèm py launcher, mở terminal mới, chạy lại `setup.bat`. Nếu Python nằm chỗ khác (pyenv, conda...), chỉ định trực tiếp: `$env:FACESLIM_PYTHON="C:\duong\dan\python.exe"; .\setup.bat` |
| `No CUDA PyTorch build worked` | Cập nhật driver NVIDIA, khởi động lại máy, chạy lại. Vẫn lỗi thì thử `.\setup.bat --cuda cu126`. |
| `CUDA GPU: not available` trong doctor | PyTorch đang là bản CPU. Chạy lại `setup.bat` (tự thay bằng bản CUDA). |
| Console in `SAM 2 teeth segmenter unavailable` | Chạy `.venv\Scripts\python tools\setup_sam2.py`, rồi `tools\doctor.py`. |
| Lỗi tải model / sai SHA-256 | File hỏng tự bị xóa; chạy `.venv\Scripts\python FaceSlim_v1.py --redownload-model sam2.1_hiera_tiny` (hoặc key khác trong `--list-models`). |
| Webcam giật khi bật nhiều hiệu ứng AI Beauty | BiSeNet đang chạy trên CPU qua ONNX Runtime. Giảm **Preview Scale** xuống 75% hoặc 50%. |
| `git` không nhận | Cài Git, mở terminal mới. |
| Video export mất tiếng | Cài FFmpeg và thêm vào `PATH`. |

Log lỗi: `crash.log` và `render.log` trong thư mục repo.

## 6. Những gì nằm ngoài git

Các thứ sau được tạo khi setup và đã được `.gitignore`, nên đừng copy từ máy khác sang:

- `.venv\`: môi trường Python
- `third_party\sam2\`: source SAM 2 (setup tự clone đúng commit)
- `*.task`, `*.onnx`, `sam2.1_*.pt`: model (setup tự tải và kiểm tra hash)

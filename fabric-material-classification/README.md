# Fabric Material AI

Project phân loại ảnh bề mặt vải thành 7 lớp bằng ResNet18 pretrained trên ImageNet và transfer learning.

## 1. Dataset

Dataset hiện có là bản export Roboflow, đặt trong `dataset/raw/` với split có sẵn. Cấu trúc:

```text
dataset/raw/{train,valid,test}/{Cotton,Cotton MIxed,Denim,Polyester,Silk,Viscose,Wool}/...
```

Xem [dataset/raw/README.md](dataset/raw/README.md). Tên lớp `Cotton MIxed` được chuẩn hóa thành `Cotton Mixed`. Dữ liệu không được commit vào Git. Project không tạo nhãn hay ảnh tổng hợp.

## 2. Cài đặt

Python 3.10 trở lên được khuyến nghị. Trên Windows/CPU, tạo môi trường ảo rồi cài PyTorch bản CPU trước, sau đó cài các dependency còn lại:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Trên máy có CUDA, thay lệnh cài PyTorch bằng lựa chọn phù hợp theo trang cài đặt chính thức của PyTorch. CPU được hỗ trợ. Lần train đầu tiên cần tải trọng số ResNet18 ImageNet pretrained từ torchvision.

## 3. Kiểm tra dữ liệu và tạo split

Chạy từ thư mục `fabric-material-classification/`:

```bash
python -m src.dataset --data dataset/raw --output manifests --seed 42
```

Lệnh kiểm tra ảnh đọc được, giữ nguyên split gốc, ghi ảnh lỗi vào `manifests/invalid_images.csv` và tạo các manifest cùng `data_summary.json`. Các biến thể Roboflow được nhóm theo tên ảnh gốc trước hậu tố `.rf.<hash>`. Nếu một nhóm xuất hiện ở nhiều split hoặc có nhãn mâu thuẫn, toàn bộ nhóm được cách ly vào `manifests/quarantined_images.csv`; ảnh gốc không bị xóa hay đổi nhãn. Báo cáo hiện tại ghi nhận 78 nhóm nguồn, 384 ảnh cần cách ly; còn 5.900 ảnh dùng được. Cách nhóm theo tên nguồn không phát hiện được mọi near-duplicate có tên khác nhau.

Không thay manifest sau khi đã bắt đầu so sánh mô hình. Không dùng test set để điều chỉnh training.

## 4. Huấn luyện

```bash
python -m src.train --epochs 20 --batch-size 32 --seed 42
```

Lượt đầu đóng băng backbone và huấn luyện classification head. Có thể fine-tune block cuối từ checkpoint tốt nhất bằng learning rate nhỏ:

```bash
python -m src.train --fine-tune --resume-checkpoint models/best_model.pth --epochs 5 --patience 2 --batch-size 32 --learning-rate 0.00001 --num-threads 4 --seed 42
```

Checkpoint tốt nhất được chọn theo validation macro F1; khi fine-tune, checkpoint cũ được giữ nếu validation không tốt hơn. Early stopping mặc định sau 5 epoch không cải thiện. Kết quả ghi vào `models/best_model.pth` và `outputs/`. Chạy huấn luyện trên CPU có thể chậm.

## 5. Đánh giá test

Chỉ chạy sau khi chốt mô hình và cấu hình:

```bash
python -m src.evaluate --manifest manifests/test.csv --checkpoint models/best_model.pth
```

Lệnh xuất `outputs/classification_report.txt` và `outputs/confusion_matrix.png`. Chỉ báo cáo metric sau khi lệnh chạy thành công; project không chứa sẵn số liệu minh họa.

### Kết quả chạy hiện tại

Sau khi cách ly các nhóm nguồn có nhãn/split mâu thuẫn, tập dữ liệu còn 5.163 ảnh train, 367 validation và 370 test. Checkpoint tốt nhất là fine-tune block cuối ở epoch 16; validation macro F1 là 0.6221. Test accuracy hiện là 0.6270 và test macro F1 là 0.6314. Denim có F1 0.8598, còn Polyester thấp nhất với F1 0.5000.

**Giới hạn phương pháp:** trong quá trình phát triển, test set đã được đánh giá một lần với checkpoint head-only trước khi quyết định thử fine-tuning. Vì thế metric test hiện tại chỉ nên xem là kết quả mô tả; để báo cáo như ước lượng độc lập, cần một test set mới chưa từng xem hoặc một lần thí nghiệm mới chỉ dùng validation để chọn mô hình.

## 6. Dự đoán một ảnh

```bash
python -m src.predict path/to/fabric.jpg
```

Hiển thị dự đoán, confidence, top-3 và xác suất của từng lớp.

## 7. Streamlit

```bash
streamlit run app.py --server.maxUploadSize 12
```

Ứng dụng yêu cầu `models/best_model.pth` đã được tạo. Nếu chưa có checkpoint, ứng dụng báo rõ trạng thái và lệnh training cần chạy. Chọn ảnh JPG/PNG tối đa 12 MB để xem preview, nhãn dự đoán, confidence, top-3 và xác suất của cả 7 lớp.

## 8. Diễn giải kết quả

- **Image classification:** dự đoán một nhãn cho toàn bộ ảnh đầu vào.
- **ResNet18:** CNN gọn, có residual connections và trọng số pretrained sẵn để phù hợp project học phần.
- **Transfer learning:** khởi đầu từ trọng số học trên ImageNet, sau đó huấn luyện head hoặc fine-tune backbone trên ảnh vải.
- **Accuracy:** tỷ lệ dự đoán đúng tổng thể.
- **Precision:** trong số dự đoán thuộc một lớp, tỷ lệ đúng.
- **Recall:** trong số ảnh thực thuộc lớp đó, tỷ lệ tìm đúng.
- **F1:** trung bình điều hòa precision và recall; macro F1 cho trọng số ngang nhau giữa các lớp.
- **Confusion matrix:** hàng là nhãn thật, cột là nhãn dự đoán; ngoài đường chéo cho thấy các lớp bị nhầm lẫn.

## 9. Cấu trúc

```text
dataset/raw/    Dataset gốc với train/valid/test, không commit
manifests/      Split cố định và báo cáo kiểm tra dữ liệu
models/         Checkpoint tốt nhất
outputs/        Training curves và đánh giá
src/dataset.py  Kiểm tra dữ liệu, chia split, transforms và Dataset
src/model.py    ResNet18
src/train.py    Huấn luyện và lưu checkpoint
src/evaluate.py Đánh giá test
src/predict.py  Dự đoán ảnh đơn
app.py          Giao diện Streamlit
```


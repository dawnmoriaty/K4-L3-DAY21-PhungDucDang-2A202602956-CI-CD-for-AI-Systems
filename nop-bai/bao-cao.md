# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phùng Đức Đăng |
| MSSV | 2A202602956 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/dawnmoriaty/K4-L3-DAY21-PhungDucDang-2A202602956-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ tham số này đạt F1-score cao nhất (0.7149), vượt qua ngưỡng kiểm định chất lượng bắt buộc (>= 0.65). Dù lần chạy 1 có accuracy nhỉnh hơn (0.8780 so với 0.8740), lần 3 tối ưu hóa việc phân loại lớp thiểu số tốt hơn, chứng minh accuracy cao nhất không đồng nghĩa với mô hình hữu ích nhất trên dữ liệu mất cân bằng. Tăng số lượng cây lên 200 cùng max_depth=5 giúp mô hình học tương tác phi tuyến tính phức tạp mà không bị thiếu khớp như lần 2 (F1 chỉ đạt 0.6051).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng lớn khi lớp dương (thu nhập > 50K) chỉ chiếm 24,8%. Một mô hình đoán toàn bộ nhãn âm vẫn đạt 75,2% accuracy nhưng hoàn toàn vô dụng do F1 bằng 0. Accuracy tạo cảm giác an toàn giả tạo vì bị chi phối bởi lớp đa số. F1-score của lớp dương là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác khả năng nhận diện đúng và không bỏ sót người có thu nhập cao. Tuyệt đối không dùng `average="weighted"` hay `"macro"` vì việc tính trung bình sẽ bị lớp 75% che lấp hiệu quả trên lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `uv pip install` báo lỗi `Invalid argument (os error 22)` | NTFS trên `/mnt/win_d` chặn thư mục `pyarrow.` kết thúc bằng dấu chấm | Chuyển `.venv` sang phân vùng Linux native (`~/.venvs/lab-21`) và tạo symlink trỏ tới |
| `mlflow` lỗi `cannot import name 'FallbackAsyncAdaptedQueuePool'` | `sqlalchemy>=2.1` gỡ class này trong khi `mlflow==2.13.0` vẫn gọi | Hạ và ghim `sqlalchemy<2.1` cùng `setuptools<72` trong `requirements.txt` |
| Quá trình build `scikit-learn` bị crash | Python mặc định trên Fedora là 3.14 chưa có wheel nhị phân | Dùng chính xác Python 3.11 (`python3.11`) để tận dụng binary wheels |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi nạp thêm 22.361 mẫu (`train_batch2`), cả F1-score và Accuracy đều tăng nhẹ (F1 tăng từ 0.7149 lên 0.7354, Accuracy từ 0.8740 lên 0.8820). Quy mô dữ liệu 44.722 mẫu giúp mô hình tinh chỉnh ranh giới phân loại lớp thiểu số tốt hơn. Quan trọng nhất, quy trình Continuous Training đã chạy hoàn toàn tự động: chỉ cần commit dữ liệu mới, pipeline tự động kéo dữ liệu, huấn luyện, vượt qua Quality Gate và deploy lên server mà không cần can thiệp thủ công.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Kết nối MLflow đến server DagsHub qua HTTPS token và ghi nhận log các thí nghiệm CI/CD.

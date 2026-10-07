# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

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

**Lý do:** Bộ tham số này đạt điểm F1-score cao nhất trên tập holdout (0.7149), vượt qua ngưỡng kiểm định chất lượng bắt buộc (F1 >= 0.65). Mặc dù lần chạy 1 có accuracy nhỉnh hơn một chút (0.8780 so với 0.8740), nhưng lần chạy 3 tối ưu hóa việc phân loại lớp thiểu số (thu nhập > 50K) tốt hơn. Điều này chứng minh lần chạy có accuracy cao nhất không nhất thiết là mô hình hữu dụng nhất trên dữ liệu mất cân bằng. Ngoài ra, việc tăng số lượng cây (n_estimators=200) kết hợp với learning_rate=0.1 và max_depth=5 giúp các cây sau bù trừ sai số tốt hơn, học được tương tác phức tạp mà không bị thiếu khớp như lần chạy 2 (F1 chỉ đạt 0.6051).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng đáng kể, trong đó lớp dương (thu nhập > 50K) chỉ chiếm khoảng 24,8% tổng số mẫu. Nếu một mô hình đơn giản luôn đoán nhãn "thu nhập thấp" cho mọi mẫu, nó vẫn đạt accuracy lên tới 75,2% nhưng hoàn toàn vô dụng vì F1-score của lớp dương bằng 0 (không phát hiện được bất kỳ ai có thu nhập cao). Do đó, accuracy tạo ra cảm giác an toàn giả tạo và bị chi phối hoàn toàn bởi lớp đa số. F1-score của lớp dương (pos_label=1) là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác khả năng mô hình vừa nhận diện đúng vừa không bỏ sót đối tượng thu nhập cao. Tuyệt đối không sử dụng `average="macro"` hoặc `average="weighted"` vì việc tính trung bình sẽ bị lớp đa số 75% kéo điểm lên, che lấp hiệu quả thực tế trên lớp thiểu số cần quan tâm.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `uv pip install` báo lỗi `Invalid argument (os error 22)` khi cài `pyarrow==15.0.2` | Thư mục wheel chứa folder rỗng `pyarrow.` (kết thúc bằng dấu chấm), bị hệ thống tệp NTFS chặn trên `/mnt/win_d` | Chuyển môi trường `.venv` sang phân vùng Linux native (`~/.venvs/lab-21`) và tạo symlink `.venv` trỏ tới đó |
| `mlflow` báo lỗi `ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool'` khi kết nối SQLite | `sqlalchemy>=2.1` vừa phát hành đã gỡ bỏ class này trong khi `mlflow==2.13.0` vẫn gọi | Hạ và ghim phiên bản `sqlalchemy<2.1` cùng `setuptools<72` tương thích trong `requirements.txt` |
| Quá trình build source `scikit-learn` bị crash | Lệnh `python3 -m venv` mặc định trên Fedora gọi Python 3.14 chưa có wheel tương thích | Sử dụng chính xác Python 3.11 (`python3.11`) để đảm bảo các wheel nhị phân có sẵn |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___

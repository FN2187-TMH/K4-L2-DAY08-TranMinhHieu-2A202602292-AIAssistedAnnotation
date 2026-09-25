# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

1. **frame_0182.jpg** (Thứ tự / Rank 1 | Điểm Score: 0.9591 | Thời điểm t = 72.8s)[cite: 64]
   - *Lý do:* Điểm tổng hợp cao nhất toàn bộ tập pool[cite: 64]. Độ bất định $U = 0.9182$ và chỉ số mơ hồ $A = 1.0$ đạt tối đa[cite: 64], chứa 28 box trong đó có 18 box mờ đục[cite: 64]. Frame này giúp thu thập thông tin chất lượng nhất về các phương tiện giao thông di chuyển ở khoảng giữa đường ban đêm[cite: 64].

2. **frame_0369.jpg** (Thứ tự / Rank 2 | Điểm Score: 0.9324 | Thời điểm t = 147.6s)[cite: 64]
   - *Lý do:* Điểm cao thứ hai, đại diện cho giai đoạn mật độ xe cực kỳ đông về cuối video ($t = 147.6s$)[cite: 64]. Có tới 43 box phát hiện với độ bất định rất cao ($U = 0.9315$)[cite: 64]. 

3. **frame_0099.jpg** (Thứ tự / Rank 8 | Điểm Score: 0.9063 | Thời điểm t = 39.6s)[cite: 64]
   - *Lý do:* Đại diện quan trọng ở khoảng thời gian đầu video ($t = 39.6s$)[cite: 64], có độ bất định $U = 0.9460$[cite: 64]. Frame này chứa xe tối màu ở gần mép lề trái đường mà mô hình pretrained COCO bỏ sót không dự đoán được box (đã được xác nhận trong `REVIEW_LOG.csv`)[cite: 64].

4. **frame_0227.jpg** (Thứ tự / Rank 11 | Điểm Score: 0.8915 | Thời điểm t = 90.8s)[cite: 64]
   - *Lý do:* Nằm ở giữa video ($t = 90.8s$), đảm bảo độ đa dạng thời gian ($D = 1.0$) và khoảng cách an toàn ($> 15s$) so với các frame được chọn khác[cite: 64]. Chứa 37 box với nhiều góc lóa đèn pha khó ($U = 0.9164$)[cite: 64].

5. **frame_0312.jpg** (Thứ tự / Rank 7 | Điểm Score: 0.9100 | Thời điểm t = 124.8s)[cite: 64]
   - *Lý do:* Thuộc khoảng thời gian $t = 124.8s$[cite: 64], chứa 37 box và có điểm mơ hồ tuyệt đối $A = 1.0$ (18 box mơ hồ)[cite: 64]. 
   - *Xét ảnh gần trùng:* **Bỏ qua `frame_0368.jpg` (Rank 9, Score 0.9003, t = 147.2s)** và **`frame_0372.jpg` (Rank 6, Score 0.9101, t = 148.8s)** vì hai ảnh này bị gần trùng thời gian ($\vert{}147.2s - 147.6s\vert{} = 0.4s < \text{MIN\_GAP\_S} = 2.0s$) với `frame_0369.jpg` đã chọn ở Rank 2[cite: 64]. Việc loại bỏ ảnh gần trùng giúp tiết kiệm ngân sách gán nhãn[cite: 64].

---

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

1. **frame_0182.jpg** ($t = 72.8s$, Rank 1, Score 0.9591): Có trong `selection_round1.csv` với cột `selected = True`[cite: 64], hiển thị ở hàng 1 ô 3 trên ảnh contact sheet `selection_round1.jpg`[cite: 64].
2. **frame_0331.jpg** ($t = 132.4s$, Rank 5, Score 0.9154): Có trong `selection_round1.csv` với cột `selected = True`[cite: 64], hiển thị ở hàng 1 ô 5 trên ảnh contact sheet `selection_round1.jpg`[cite: 64].
3. **frame_0392.jpg** ($t = 156.8s$, Rank 15, Score 0.8874): Có trong `selection_round1.csv` với cột `selected = True`[cite: 64], hiển thị ở hàng 2 ô 6 trên ảnh contact sheet `selection_round1.jpg`[cite: 64].

---

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

- **Frame có điểm cao nhưng KHÔNG được chọn:** `frame_0372.jpg` (Thứ tự Rank 6 | Điểm Score = 0.9101 | $t = 148.8s$)[cite: 64].
  - *Lý do:* Mặc dù thuộc top 6 điểm cao nhất[cite: 64], frame này bị thuật toán gạt bỏ (`selected = False`) do cách `frame_0369.jpg` (Rank 2) chỉ 1.2 giây ($\vert{}148.8s - 147.6s\vert{} = 1.2s < 2.0s$)[cite: 64]. Do bối cảnh giao thông gần như không đổi, việc gán nhãn cả 2 frame sẽ gây lãng phí công sức mà không mang lại thêm thông tin mới cho mô hình[cite: 64].

---

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

- **Chưa chứng minh mô hình sẽ học tốt hơn sau khi huấn luyện:** Phép chọn dựa vào Active Learning (Uncertainty & Ambiguity) chỉ giúp lọc ra các ảnh "khó" hoặc "mô hình đang phân vân"[cite: 64]. Điểm bất định cao không đảm bảo rằng khi gán nhãn và fine-tune lại thì mô hình sẽ tăng chỉ số mAP/AP50 trên tập test[cite: 64]. Trong thực tế Vòng 1, do số lượng ảnh train còn quá ít (12 ảnh), mô hình fine-tune đã bị suy giảm chỉ số Recall và mAP so với Cold Start do hiện tượng overfitting/thận trọng quá mức[cite: 64].
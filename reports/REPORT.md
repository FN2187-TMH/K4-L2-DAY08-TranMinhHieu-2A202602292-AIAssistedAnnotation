# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trần Minh Hiếu

Công cụ gán nhãn đã dùng: Sửa trực tiếp file nhãn (kèm công cụ gán nhãn CVAT

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

**Trả lời:**

- **Tại sao chia theo trục thời gian và có vùng đệm:**
  Dữ liệu video ban đêm có tính phụ thuộc thời gian cao (temporal correlation) giữa các frame liên tiếp (cùng bối cảnh đường xá, góc quay camera, điều kiện ánh sáng và hành trình xe di chuyển).
  1. Việc chia theo trục thời gian phản ánh đúng bài toán thực tế (mô hình huấn luyện trên dữ liệu quá khứ và phải dự đoán tốt trên video tương lai).
  2. Vùng đệm thời gian (time buffer) ở giữa giúp triệt tiêu hoàn toàn sự rò rỉ dữ liệu (data leakage) giữa tập pool và tập test, tránh việc các frame thuộc tập test quá giống các frame nằm sát lề tập pool.

- **Nếu chia ngẫu nhiên, số đo bị lệch theo hướng nào và vì sao:**
  Nếu chia ngẫu nhiên (random split), các frame trong tập test và tập pool/train sẽ rất giống nhau về bối cảnh, góc xe và vệt sáng đèn. Điều này làm xảy ra hiện tượng **data leakage**, dẫn đến việc số đo đánh giá (mAP, Precision, Recall) trên tập test bị **lệch cao hơn thực tế (over-optimistic / bốc đồng)**. Mô hình chỉ "học thuộc lòng" các bối cảnh trùng lặp thay vì có khả năng tổng quát hóa (generalization) tốt.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

**Trả lời:**

- **Dòng vòng 0 chép từ `rounds_table.md`:**

| Vòng | Mô hình | Ảnh train | Box train | Chiến lược | AP50 | Precision | Recall | Recall (small) | Recall (medium) | Recall (large) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cold start (COCO) | 0 | 0 | - | 0.771 | 0.925 | 0.489 | 0.1818 | 0.5473 | 0.5610 |

- **Sự không khớp nhãn tham chiếu ở mô hình cold start:**
  Mô hình `yolov8n` cold start (pretrained trên COCO) không khớp nhãn ở các loại xe nằm ở xa (xe kích thước nhỏ ở sát đường chân trời), xe ban đêm chỉ có vệt đèn pha mờ/tối, hoặc các xe bị khuất lấp/xe di chuyển ở các làn đường phía lề trái/phía xa.

- **Ý nghĩa của độ phủ (Recall) theo kích thước xe:**
  - `small`: 0.1818 (18.18%)
  - `medium`: 0.5473 (54.73%)
  - `large`: 0.5610 (56.10%)
  
  Con số này cho thấy mô hình cold start bỏ sót nghiêm trọng đối với **xe kích thước nhỏ (Recall chỉ 18.18%)** do ánh sáng ban đêm mờ và xe ở xa bị biến dạng, trong khi với xe kích thước trung bình và lớn đạt độ phủ tốt hơn (>54%).

- **Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:**
  Trường hợp mô hình phát hiện đúng một chiếc xe trong đêm (có đèn hoặc có thân xe rõ ràng) nhưng trong tập nhãn tham chiếu (ground truth / reference) người gán nhãn ban đầu đánh sót (chưa gán box), hoặc trường hợp xe nhỏ hơn ngưỡng quy định (dưới 16 pixel) nhưng vẫn bị mô hình bắt box. Lúc này box phát hiện bị tính là False Positive (FP), do đó cần rà soát thủ công nhãn tham chiếu để đảm bảo công bằng.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

**Trả lời:**

- **Giải thích công thức `score = W_U·U + W_A·A + W_D·D`:**
  - `U` (Uncertainty): Độ bất định của dự đoán trên frame (dựa trên điểm tin cậy conf của các box gần ngưỡng mờ đục 0.5).
  - `A` (Ambiguous/Ambiguity): Số lượng box mơ hồ trong ảnh (ảnh chứa nhiều đối tượng khó nhận diện/mờ nhòe).
  - `D` (Diversity / Distance): Độ đa dạng theo thời gian, ưu tiên các ảnh xa các ảnh đã chọn để tránh lặp bối cảnh.
  - `W_U, W_A, W_D`: Các trọng số gán cho từng tiêu chí (ở đây cấu hình lần lượt là `0.5`, `0.3`, `0.2`). Điểm tổng hợp `score` giúp chọn ra lô ảnh vừa khó đối với mô hình, vừa chứa nhiều đối tượng mờ nhòe và vừa đảm bảo tính đa dạng.

- **Vai trò của `MIN_GAP_S`:**
  `MIN_GAP_S = 2.0` (giây) đặt ra khoảng cách thời gian tối thiểu giữa hai frame được chọn trong cùng một lô (batch). Vai trò của nó là loại bỏ các ảnh gần như trùng lặp hoàn toàn (near-duplicate) do tần số lấy mẫu video cao, giúp tiết kiệm công sức gán nhãn và tối ưu hiệu quả thông tin.

- **Minh chứng chọn mẫu từ các frame:**
  - `frame_0000.jpg` ($t=0.0s$, score high): Đại diện frame đầu chuỗi video, có độ đa dạng thời gian cao và chứa nhiều xe ở làn xa.
  - `frame_0074.jpg` ($t=29.6s$): Frame có mật độ giao thông đông, nhiều vệt đèn pha gây nhiễu, độ bất định $U$ cao.
  - `frame_0129.jpg` ($t=51.6s$): Frame có nhiều xe kích thước trung bình và nhỏ ở khoảng giữa đường, tối và mơ hồ.
  - `frame_0200.jpg` ($t=80.0s$): Minh chứng tính toán khoảng cách `MIN_GAP_S` loại bỏ các frame liền kề như `frame_0206.jpg` ($t=82.4s$) dù có độ bất định tương đương, tránh lãng phí công gán nhãn.

- **Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Vì sao?**
  **Không.** Điểm bất định cao chỉ thể hiện mô hình hiện tại đang phân vân/thiếu tin tưởng vào dự đoán của nó trên ảnh đó. Nếu ảnh bị bất định do nhiễu nặng (out-of-distribution noise, lóa đèn pha quá mức, ống kính mờ) hoặc nhãn bị gán sai/mơ hồ (ambiguous label), việc đưa thêm ảnh này vào huấn luyện có thể khiến mô hình bị nhiễu (label noise) và thậm chí làm giảm độ chính xác chung.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

**Trả lời:**

- **Bảng tổng hợp chép từ `rounds_table.md`:**

| Vòng | Mô hình | Ảnh train | Box train | Chiến lược | AP50 | Precision | Recall | Recall (small) | Recall (medium) | Recall (large) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | cold start (COCO) | 0 | 0 | - | 0.771 | 0.925 | 0.489 | 0.1818 | 0.5473 | 0.5610 |
| 1 | fine-tune vong 1..1 | 12 | 317 | uncertainty | 0.386 | 1.000 | 0.015 | 0.0000 | 0.0150 | 0.0200 |

- **Phân tích chi tiết Vòng 1:**
  - **Mức độ sửa nhãn gợi ý:** Trong Vòng 1 (lô 12 ảnh), người gán nhãn đã thực hiện sửa đổi nhãn gợi ý (pre-labels):
    - Số box giữ nguyên: 302
    - Số box thêm mới: 15 (ví dụ: các xe tối gần mép đường trong `REVIEW_LOG.csv` như ở `frame_0099.jpg`)
    - Số box chỉnh sửa / xóa: Sửa lại các ranh giới box sát thân xe theo `GUIDELINE_LABEL.md`.
    - Tổng số box sau gán nhãn: 317 box / 12 ảnh.
  - **Mức độ thay đổi AP50:**
    - So với khởi đầu lạnh (Cold Start): AP50 **giảm mạnh 0.385** (từ 0.771 xuống 0.386).
    - Precision tăng lên 1.000 nhưng Recall giảm thảm hại từ 0.489 xuống 0.015.
  - **Nhóm xe tốt lên / xấu đi:**
    - Toàn bộ các nhóm xe (`small`, `medium`, `large`) đều bị suy giảm Recall trầm trọng (đặc biệt xe `small` giảm về 0.0000) do mô hình bị overfit nặng vào tập dữ liệu huấn luyện quá nhỏ (12 ảnh) và thiếu sự đa dạng mẫu ban đầu.

- **Ca kết quả thay đổi sau fine-tune và nguyên nhân:**
  Dựa vào `outputs/compare_round1.jpg`:
  - Quan sát tại `frame_0050` & `frame_0150`: Mô hình Vòng 1 bỏ sót hầu hết các xe (số lượng False Negative - FN tăng vọt, thể hiện qua rất nhiều box màu vàng).
  - **Lý do kiểm chứng:** Việc fine-tune 50 epochs chỉ với 12 ảnh (317 box) từ mô hình pretrained COCO làm thay đổi lớp head phân loại, gây ra hiện tượng **catastrophic forgetting** (quên tri thức cũ) và suy giảm mạnh ngưỡng dự đoán tin cậy trên dữ liệu mới khi chưa đủ số lượng ảnh phong phú.

- **Phân biệt quan sát, lỗi pre-label và kết quả sau train:**
  - *Quan sát độc lập (`BLIND_SCAN.md`)*: Phát hiện các xe bị che khuất một phần hoặc xe tối ở lề trái/phải đường trong các frame ban đêm.
  - *Lỗi pre-label đã sửa (`REVIEW_LOG.csv` & `round1_diff.md`)*: Pre-label do mô hình gợi ý bỏ sót xe tối lề trái tại `frame_0099.jpg`; người gán nhãn đã chủ động `added` (thêm mới) box theo đúng quy định.
  - *Kết quả mô hình sau train*: Mô hình Vòng 1 trở nên cực kỳ thận trọng (Precision 1.0), chỉ bắt những box thật sự chắc chắn và bỏ qua các box có độ tin cậy thấp hơn.

- **Ca khó theo guideline:**
  Trường hợp hai xe ô tô chạy sát nhau theo chiều dọc hoặc xe bị lóa đèn pha chiếu thẳng vào camera: Nhãn gợi ý (pre-label) gộp chung hai xe thành một box lớn. Theo `GUIDELINE_LABEL.md`, gán nhãn viên phải tách thành hai box riêng biệt dựa vào thân xe và vị trí hai cặp đèn pha độc lập.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

**Trả lời:**

- **Kết quả so với Cold Start & Đánh giá việc dừng/tiếp tục:**
  - Kết quả Vòng 1 (AP50 = 0.386) thấp hơn đáng kể so với Cold Start (AP50 = 0.771).
  - **Quyết định:** **Nên tiếp tục (Không dừng lại)**. Việc AP50 giảm ở vòng 1 là hiện tượng phổ biến khi mới chỉ gán nhãn 12 ảnh (lô đầu tiên), làm mô hình bị hẫng do số lượng mẫu huấn luyện quá ít. Cần tiếp tục các vòng Active Learning tiếp theo (Vòng 2, Vòng 3 với lô ảnh `selection_round2.csv`) để bổ sung dữ liệu huấn luyện giúp mô hình hồi phục và vượt điểm cold start.

- **Đề xuất 2 ca còn yếu/bất định cho vòng sau:**
  1. **Ca xe nhỏ ở sát đường chân trời (far-distance vehicles):** Độ phủ `small` đang là 0.0%. Chi phí rà nhãn cao do phải zoom kĩ để gán, nguy cơ ảnh gần trùng cao nếu chọn các frame liên tiếp.
  2. **Ca xe bị chói đèn pha/lóa sáng ban đêm (headlight glare):** Mức độ bất định cao do vệt sáng che mất thân xe. Chi phí rà nhãn trung bình, nguy cơ trùng lặp bối cảnh nếu không kiểm soát `MIN_GAP_S`.

- **Ảnh hưởng từ các giới hạn của tập kiểm thử:**
  - Tập test nhỏ (chỉ 20 ảnh) có thể gây ra độ biến động (variance) cao trong chỉ số mAP.
  - Luật bỏ qua xe quá nhỏ (<16px) làm số đo mAP phụ thuộc vào cách lọc kích thước.
  - Nhãn tham chiếu chưa được rà thủ công toàn bộ có thể chứa lỗi nhãn ẩn (ground truth noise), dẫn đến việc đánh giá mô hình chưa phản ánh đúng 100% thực tế.

- **Các điểm cần kiểm tra khi AP50 giảm trước khi train thêm:**
  1. **Kiểm tra chất lượng gán nhãn (Label Quality & Noise):** Kiểm tra xem trong các file nhãn đã sửa có box nào bị lệch, gán nhãn sai lớp hay bỏ sót đối tượng rõ ràng không.
  2. **Kiểm tra cấu hình và Hyperparameters:** Kiểm tra Learning Rate, số lượng Epochs (50 epochs với 12 ảnh có thể bị overfit), và cơ chế Freeze Layers/Unfreeze Layers khi fine-tune từ `yolov8n.pt`.
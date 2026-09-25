# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 22

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe taxi màu vàng/cam ở làn bên phải trung cảnh: Xe bị tối phần thân phía sau và đèn hậu không rực sáng, AI dễ bỏ sót do không có vệt đèn phản chiếu mạnh như các xe đi ngược chiều/đèn pha sáng.
2. Các xe phía xa ở làn đường chính và xe ở góc dưới bên phải: Xe ở xa chỉ hiển thị dưới dạng hai đốm đèn đỏ/trắng nhỏ (dễ bị AI nhầm với ánh đèn phản chiếu trên mặt đường hoặc vệt sáng giao thông), còn xe góc dưới bên phải bị cắt xẻng một phần ở mép ảnh.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
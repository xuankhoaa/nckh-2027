# Báo cáo NCKH dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Selective State Space Model

## Tóm tắt

Nghiên cứu xây dựng quy trình dự báo log return một phiên kế tiếp cho tám cổ phiếu ngân hàng Việt Nam. Dữ liệu OHLCV và VN-Index trong giai đoạn 2020-2026 được chia theo thời gian thành tập huấn luyện, xác thực và kiểm tra. Nghiên cứu đối chiếu hai baseline hằng số, LSTM, Selective SSM cơ sở và Selective SSM có thêm return VN-Index. Mọi cấu hình mạng được lặp trên năm seed; kết quả được đánh giá bằng RMSE, MAE, R², độ chính xác hướng và kiểm định chênh lệch squared loss. Kết quả chi tiết được sinh tự động từ mã nguồn và cần được diễn giải trong phạm vi snapshot dữ liệu đã công bố.

**Từ khóa:** dự báo tỷ suất sinh lợi; cổ phiếu ngân hàng; Selective State Space Model; Mamba; LSTM; thị trường chứng khoán Việt Nam.

## 1. Mở đầu

Dự báo return ngắn hạn là bài toán hồi quy có độ nhiễu cao. Mục tiêu của nghiên cứu không phải đưa ra khuyến nghị đầu tư mà là đánh giá công bằng khả năng dự báo của các kiến trúc chuỗi thời gian trên cùng dữ liệu và cùng quy trình không rò rỉ dữ liệu. Câu hỏi nghiên cứu là liệu Selective SSM có giảm sai số dự báo so với baseline và LSTM, và việc thêm return VN-Index có hỗ trợ mô hình hay không.

## 2. Dữ liệu và phương pháp

Đầu vào gồm open, high, low, close, volume; các đặc trưng dẫn xuất là log return, thay đổi khối lượng, biên độ high-low chuẩn hóa và log return VN-Index. Nhãn là `log(close_(t+1) / close_t)`. Mỗi mẫu dùng 30 phiên kết thúc tại t để dự báo t+1. StandardScaler chỉ fit trên các quan sát thuộc train. Train kéo dài đến 31/12/2023, validation là năm 2024, test bắt đầu từ 01/01/2025.

Các mạng dùng hidden size 8, một lớp, Adam với learning rate 0,001, tối đa 40 epoch và early stopping sau 8 epoch không cải thiện trên validation. Nhãn được chuẩn hóa bằng trung bình và độ lệch chuẩn của train, rồi đảo chuẩn hóa trước đánh giá. LSTM có 585 tham số; Selective SSM cơ sở có 545 tham số và bản thêm VN-Index có 553 tham số. Selective SSM được tự triển khai từ mô tả thuật toán Mamba; mã nguồn MambaStock không được đưa vào kho.

## 3. Kết quả

| Mô hình                   | Cửa sổ phiên | Tập đặc trưng                                | Số mã | Số seed | Số tham số | RMSE trung bình | MAE trung bình | R² trung bình | DA trung bình | ĐLC RMSE giữa mã | ĐLC MAE giữa mã | ĐLC R² giữa mã | ĐLC DA giữa mã | ĐLC RMSE giữa seed | ĐLC MAE giữa seed | ĐLC R² giữa seed | ĐLC DA giữa seed |
| ------------------------- | ------------ | -------------------------------------------- | ----- | ------- | ---------- | --------------- | -------------- | ------------- | ------------- | ---------------- | --------------- | -------------- | -------------- | ------------------ | ----------------- | ---------------- | ---------------- |
| Baseline zero return      | 30           | Không áp dụng                                | 8     | 1       | 0          | 0.018928        | 0.012859       | -0.001364     | 0.078588      | 0.002384         | 0.002028        | 0.001438       | 0.028659       | 0.000000           | 0.000000          | 0.000000         | 0.000000         |
| Baseline trung bình train | 30           | Không áp dụng                                | 8     | 1       | 0          | 0.018918        | 0.012899       | -0.000482     | 0.462415      | 0.002370         | 0.002005        | 0.000441       | 0.033765       | 0.000000           | 0.000000          | 0.000000         | 0.000000         |
| LSTM                      | 30           | OHLCV và đặc trưng dẫn xuất                  | 8     | 5       | 585        | 0.019195        | 0.013171       | -0.030056     | 0.459055      | 0.002443         | 0.002043        | 0.013938       | 0.018197       | 0.000323           | 0.000339          | 0.036557         | 0.021752         |
| Selective SSM cơ sở       | 30           | OHLCV và đặc trưng dẫn xuất                  | 8     | 5       | 545        | 0.019145        | 0.013180       | -0.024805     | 0.459510      | 0.002390         | 0.002023        | 0.008905       | 0.019179       | 0.000155           | 0.000189          | 0.016868         | 0.022782         |
| Selective SSM có VN-Index | 30           | OHLCV, đặc trưng dẫn xuất và return VN-Index | 8     | 5       | 553        | 0.019156        | 0.013211       | -0.026057     | 0.461048      | 0.002433         | 0.002066        | 0.020065       | 0.016982       | 0.000354           | 0.000375          | 0.038090         | 0.022012         |

Kết quả theo từng mã, từng seed, dự báo theo ngày và kiểm định HAC nằm trong các CSV đi kèm. Mô hình có RMSE trung bình thấp nhất là **Baseline trung bình train**. Mạng tốt nhất theo RMSE là **Selective SSM cơ sở** (0.019145), nhưng vẫn cao hơn baseline trung bình train (0.018918) 0.000227. Vì vậy, câu trả lời trung thực cho câu hỏi nghiên cứu trong cấu hình hiện tại là: chưa có bằng chứng rằng mạng Selective SSM hoặc LSTM cải thiện sai số trung bình nhóm so với baseline hằng. Thêm return VN-Index cũng không giảm RMSE so với Selective SSM cơ sở (chênh +0.000011). Kết luận này không khẳng định ưu thế tổng quát ngoài khoảng dữ liệu và thiết lập đã kiểm tra.

## 4. Thảo luận và giới hạn

Return một phiên chịu ảnh hưởng của thông tin mới và nhiễu thị trường, nên R² âm cần được báo cáo minh bạch. Độ chính xác hướng không thay thế sai số hồi quy. Nghiên cứu chưa dùng dữ liệu tin tức, sentiment, yếu tố vĩ mô, báo cáo tài chính hoặc nhãn biến động/rủi ro. Dữ liệu VN-Index có cảnh báo nguồn được ghi trong manifest; pipeline chỉ dùng close để tính return và không tự thay đổi số liệu gốc.

## 5. Kết luận

Mã nguồn cung cấp quy trình tái lập từ dữ liệu, kiểm tra chất lượng, tạo đặc trưng, baseline, huấn luyện nhiều seed, lưu manifest và sinh báo cáo. Kết quả hiện tại là một kết quả âm có giá trị: các feature và cấu hình đang dùng chưa vượt baseline train mean ở trung bình nhóm. Hướng mở rộng là walk-forward validation, thêm đặc trưng ngoài giá-khối lượng, dự báo độ biến động tương lai và phân loại rủi ro.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Shi, Z. (2024). MambaStock: Selective State Space Model for Stock Prediction. arXiv:2402.18959.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.

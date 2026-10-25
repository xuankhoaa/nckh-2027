# Báo cáo kết quả hai bài toán

Mỗi mạng được so sánh trên cùng tập đặc trưng. Train Mean là ngưỡng tối thiểu; Market-only và Market+News là thí nghiệm loại bỏ đặc trưng, không phải tên mô hình.

## Tỷ suất sinh lợi phiên kế tiếp (horizon=1)

| Mô hình                     | Tập đặc trưng | Số mã | Số seed | RMSE     | MAE      | R² OOS    | DA       |
| --------------------------- | ------------- | ----- | ------- | -------- | -------- | --------- | -------- |
| LSTM                        | Market + News | 8     | 5       | 0.019228 | 0.013224 | -0.032925 | 0.455752 |
| LSTM                        | Market-only   | 8     | 5       | 0.019319 | 0.013368 | -0.044754 | 0.463895 |
| Selective SSM (Mamba-style) | Market + News | 8     | 5       | 0.019127 | 0.013184 | -0.022165 | 0.459226 |
| Selective SSM (Mamba-style) | Market-only   | 8     | 5       | 0.019081 | 0.013147 | -0.017150 | 0.456720 |
| Baseline return bằng 0      | Không áp dụng | 8     | 1       | 0.018928 | 0.012859 | -0.000815 | 0.078588 |
| Baseline trung bình train   | Không áp dụng | 8     | 1       | 0.018918 | 0.012904 | 0.000000  | 0.462415 |

Cấu hình RMSE thấp nhất: **Baseline trung bình train / Không áp dụng** (0.018918). Mạng tốt nhất chưa vượt Train Mean (0.018918).

## Độ biến động thực hiện tương lai (horizon=5)

| Mô hình                     | Tập đặc trưng | Số mã | Số seed | RMSE     | MAE      | R² OOS   | QLIKE    |
| --------------------------- | ------------- | ----- | ------- | -------- | -------- | -------- | -------- |
| LSTM                        | Market + News | 8     | 5       | 0.010110 | 0.007536 | 0.079817 | 0.834817 |
| LSTM                        | Market-only   | 8     | 5       | 0.009871 | 0.007530 | 0.125247 | 0.728266 |
| Selective SSM (Mamba-style) | Market + News | 8     | 5       | 0.009905 | 0.007624 | 0.118645 | 0.761999 |
| Selective SSM (Mamba-style) | Market-only   | 8     | 5       | 0.009779 | 0.007554 | 0.142239 | 0.683595 |
| Baseline trung bình train   | Không áp dụng | 8     | 1       | 0.010576 | 0.008696 | 0.000000 | 0.692187 |

Cấu hình RMSE thấp nhất: **Selective SSM (Mamba-style) / Market-only** (0.009779). Mạng tốt nhất vượt Train Mean (0.010576).

## Cách đọc kết quả

RMSE/MAE càng thấp càng tốt. DA chỉ dùng cho return; QLIKE chỉ dùng cho volatility. Các kiểm định HAC trong `KIEM_DINH_HAC_VI.csv` so sánh từng seed mạng với Train Mean; giá trị cải thiện MSE dương mới có lợi cho mạng. Kết luận phải đồng thời xem từng mã, trung bình nhóm và độ lệch chuẩn giữa seed.

## Giá trị tăng thêm của tin tức

| Bài toán                         | Mô hình                     | Số cặp mã-seed | Số cặp news tốt hơn | Số cặp tốt hơn với p<0,05 | Cải thiện MSE trung bình |
| -------------------------------- | --------------------------- | -------------- | ------------------- | ------------------------- | ------------------------ |
| Tỷ suất sinh lợi phiên kế tiếp   | LSTM                        | 40             | 16                  | 6                         | 0.00000371               |
| Tỷ suất sinh lợi phiên kế tiếp   | Selective SSM (Mamba-style) | 40             | 18                  | 6                         | -0.00000172              |
| Độ biến động thực hiện tương lai | LSTM                        | 40             | 14                  | 3                         | -0.00000503              |
| Độ biến động thực hiện tương lai | Selective SSM (Mamba-style) | 40             | 18                  | 3                         | -0.00000266              |

Bảng này ghép đúng cùng mô hình, mã và seed. Cải thiện MSE dương nghĩa là Market+News tốt hơn Market-only; p-value HAC là bằng chứng bổ sung, chưa điều chỉnh nhiều lần.

Manifest lần chạy: Git `e41c75dee1a3fc3a1a6f2f91cb6a93dc368db586`, Python 3.13.13.

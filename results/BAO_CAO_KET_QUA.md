# Báo cáo kết quả hai bài toán

Mỗi mô hình được so sánh trên cùng tập đặc trưng. Train Mean là ngưỡng tối thiểu; Ridge là baseline tuyến tính chọn mức phạt bằng validation; Market-only và Market+News là thí nghiệm loại bỏ đặc trưng, không phải tên mô hình.

## Tỷ suất sinh lợi phiên kế tiếp

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS    | DA       |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | --------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.018954 | 0.012962 | -0.003499 | 0.466629 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.018964 | 0.012967 | -0.004419 | 0.469476 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.018934 | 0.012939 | -0.001434 | 0.467084 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.018923 | 0.012929 | -0.000126 | 0.463952 |
| Baseline return bằng 0             | Không áp dụng            | 8     | 1       | 0.018928 | 0.012859 | -0.000528 | 0.078588 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.018936 | 0.012936 | -0.001688 | 0.465831 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.018939 | 0.012954 | -0.002257 | 0.460421 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.018921 | 0.012921 | 0.000000  | 0.462415 |

Cấu hình RMSE thấp nhất: **Baseline trung bình train / Không áp dụng** (0.018921). Mạng tốt nhất chưa vượt Train Mean (0.018921).

## Tỷ suất sinh lợi tích lũy 5 phiên

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS    | DA       |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | --------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.042368 | 0.029718 | -0.010051 | 0.516287 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.042251 | 0.029658 | -0.004272 | 0.524601 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.042262 | 0.029652 | -0.004833 | 0.520501 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.042208 | 0.029654 | -0.002362 | 0.517084 |
| Baseline return bằng 0             | Không áp dụng            | 8     | 1       | 0.042214 | 0.029523 | -0.001933 | 0.000000 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.042462 | 0.030006 | -0.014534 | 0.507403 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.042363 | 0.029833 | -0.010069 | 0.519362 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.042155 | 0.029591 | 0.000000  | 0.520216 |

Cấu hình RMSE thấp nhất: **Baseline trung bình train / Không áp dụng** (0.042155). Mạng tốt nhất chưa vượt Train Mean (0.042155).

## Độ biến động thực hiện 5 phiên tương lai

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS   | QLIKE    |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | -------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.009671 | 0.007276 | 0.147791 | 0.689216 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.009756 | 0.007197 | 0.132772 | 0.759970 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.009772 | 0.007333 | 0.128632 | 0.730341 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.009829 | 0.007278 | 0.118622 | 0.784686 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.010239 | 0.007208 | 0.046680 | 0.862484 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.010447 | 0.007280 | 0.007350 | 1.092153 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.010491 | 0.008567 | 0.000000 | 0.692076 |

Cấu hình RMSE thấp nhất: **LSTM / Market stationary** (0.009671). Mạng tốt nhất vượt Train Mean (0.010491).

## Cách đọc kết quả

RMSE/MAE càng thấp càng tốt. DA chỉ dùng cho return; QLIKE chỉ dùng cho volatility. Các kiểm định HAC trong `KIEM_DINH_HAC_VI.csv` so sánh từng lần chạy Ridge/mạng với Train Mean; giá trị cải thiện MSE dương mới có lợi cho mô hình. Hệ số co dự báo được ước lượng chỉ trên validation, không dùng test. Kết luận phải đồng thời xem từng mã, trung bình nhóm và độ lệch chuẩn giữa seed.

## Giá trị tăng thêm của tin tức

| Bài toán                                 | Horizon | Mô hình                            | Số cặp mã-seed | Số cặp news tốt hơn | Số cặp tốt hơn với p<0,05 | Cải thiện MSE trung bình |
| ---------------------------------------- | ------- | ---------------------------------- | -------------- | ------------------- | ------------------------- | ------------------------ |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | LSTM                               | 40             | 19                  | 2                         | -0.00000049              |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Selective SSM (Mamba-style)        | 40             | 20                  | 2                         | 0.00000038               |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Ridge (alpha chọn bằng validation) | 8              | 3                   | 0                         | -0.00000001              |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | LSTM                               | 40             | 17                  | 2                         | 0.00000998               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Selective SSM (Mamba-style)        | 40             | 19                  | 3                         | 0.00000474               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Ridge (alpha chọn bằng validation) | 8              | 6                   | 0                         | 0.00000910               |
| Độ biến động thực hiện 5 phiên tương lai | 5       | LSTM                               | 40             | 16                  | 0                         | -0.00000183              |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Selective SSM (Mamba-style)        | 40             | 17                  | 0                         | -0.00000121              |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Ridge (alpha chọn bằng validation) | 8              | 2                   | 0                         | -0.00000479              |

Bảng này ghép đúng cùng mô hình, mã và seed. Cải thiện MSE dương nghĩa là Market+News tốt hơn Market-only; p-value HAC là bằng chứng bổ sung, chưa điều chỉnh nhiều lần.

Manifest lần chạy: Git `b5486f6e4d3c68deb899d87ca41b0c2ceca80923`, Python 3.13.13.

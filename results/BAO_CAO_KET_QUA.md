# Báo cáo kết quả hai bài toán

Mỗi mô hình được so sánh trên cùng tập đặc trưng. Train Mean là ngưỡng tối thiểu; Ridge là baseline tuyến tính chọn mức phạt bằng validation; Market-only và Market+News là thí nghiệm loại bỏ đặc trưng, không phải tên mô hình.

## Tỷ suất sinh lợi phiên kế tiếp

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS    | DA       |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | --------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.018954 | 0.012962 | -0.003499 | 0.466629 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.018959 | 0.012971 | -0.003839 | 0.466059 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.018934 | 0.012939 | -0.001434 | 0.467084 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.018930 | 0.012934 | -0.000890 | 0.466002 |
| Baseline return bằng 0             | Không áp dụng            | 8     | 1       | 0.018928 | 0.012859 | -0.000528 | 0.078588 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.018936 | 0.012936 | -0.001688 | 0.465831 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.018939 | 0.012953 | -0.002253 | 0.464408 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.018921 | 0.012921 | 0.000000  | 0.462415 |

Cấu hình RMSE thấp nhất: **Baseline trung bình train / Không áp dụng** (0.018921). Mạng tốt nhất chưa vượt Train Mean (0.018921).

## Tỷ suất sinh lợi tích lũy 5 phiên

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS    | DA       |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | --------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.042368 | 0.029718 | -0.010051 | 0.516287 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.042259 | 0.029688 | -0.004518 | 0.522096 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.042262 | 0.029652 | -0.004833 | 0.520501 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.042209 | 0.029651 | -0.002417 | 0.518280 |
| Baseline return bằng 0             | Không áp dụng            | 8     | 1       | 0.042214 | 0.029523 | -0.001933 | 0.000000 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.042462 | 0.030006 | -0.014534 | 0.507403 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.042208 | 0.029666 | -0.002344 | 0.519077 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.042155 | 0.029591 | 0.000000  | 0.520216 |

Cấu hình RMSE thấp nhất: **Baseline trung bình train / Không áp dụng** (0.042155). Mạng tốt nhất chưa vượt Train Mean (0.042155).

## Độ biến động thực hiện 5 phiên tương lai

| Mô hình                            | Tập đặc trưng            | Số mã | Số seed | RMSE     | MAE      | R² OOS   | QLIKE    |
| ---------------------------------- | ------------------------ | ----- | ------- | -------- | -------- | -------- | -------- |
| LSTM                               | Market stationary        | 8     | 5       | 0.009671 | 0.007276 | 0.147791 | 0.689216 |
| LSTM                               | Market stationary + News | 8     | 5       | 0.009776 | 0.007209 | 0.129236 | 0.761927 |
| Selective SSM (Mamba-style)        | Market stationary        | 8     | 5       | 0.009772 | 0.007333 | 0.128632 | 0.730341 |
| Selective SSM (Mamba-style)        | Market stationary + News | 8     | 5       | 0.009826 | 0.007293 | 0.119319 | 0.783877 |
| Ridge (alpha chọn bằng validation) | Market stationary        | 8     | 1       | 0.010239 | 0.007208 | 0.046680 | 0.862484 |
| Ridge (alpha chọn bằng validation) | Market stationary + News | 8     | 1       | 0.010459 | 0.007289 | 0.004374 | 1.153182 |
| Baseline trung bình train          | Không áp dụng            | 8     | 1       | 0.010491 | 0.008567 | 0.000000 | 0.692076 |

Cấu hình RMSE thấp nhất: **LSTM / Market stationary** (0.009671). Mạng tốt nhất vượt Train Mean (0.010491).

## Cách đọc kết quả

RMSE/MAE càng thấp càng tốt. DA chỉ dùng cho return; QLIKE chỉ dùng cho volatility. Các kiểm định HAC trong `KIEM_DINH_HAC_VI.csv` so sánh từng lần chạy Ridge/mạng với Train Mean; giá trị cải thiện MSE dương mới có lợi cho mô hình. Hệ số co dự báo được ước lượng chỉ trên validation, không dùng test. Kết luận phải đồng thời xem từng mã, trung bình nhóm và độ lệch chuẩn giữa seed.

## So sánh trực tiếp Mamba-style với LSTM

| Bài toán                                 | Horizon | Tập đặc trưng            | Số cặp mã-seed | Số cặp Mamba tốt hơn | Cải thiện MSE trung bình |
| ---------------------------------------- | ------- | ------------------------ | -------------- | -------------------- | ------------------------ |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Market stationary        | 40             | 21                   | 0.00000076               |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Market stationary + News | 40             | 25                   | 0.00000120               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Market stationary        | 40             | 16                   | 0.00000922               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Market stationary + News | 40             | 14                   | 0.00000504               |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Market stationary        | 40             | 12                   | -0.00000194              |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Market stationary + News | 40             | 13                   | -0.00000084              |

Mỗi cặp dùng cùng mã, tập đặc trưng, seed và ngày test. Cải thiện MSE dương có lợi cho Mamba-style. Các seed trên cùng mã dùng chung chuỗi test, nên số cặp không phải số quan sát độc lập; xem `architecture_tests.csv` để đọc kiểm định HAC từng cặp, chưa điều chỉnh nhiều lần.

## Giá trị tăng thêm của tin tức

| Bài toán                                 | Horizon | Mô hình                            | Số cặp mã-seed | Số cặp news tốt hơn | Số cặp tốt hơn với p<0,05 | Cải thiện MSE trung bình |
| ---------------------------------------- | ------- | ---------------------------------- | -------------- | ------------------- | ------------------------- | ------------------------ |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | LSTM                               | 40             | 19                  | 3                         | -0.00000031              |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Selective SSM (Mamba-style)        | 40             | 19                  | 2                         | 0.00000013               |
| Tỷ suất sinh lợi phiên kế tiếp           | 1       | Ridge (alpha chọn bằng validation) | 8              | 4                   | 0                         | -0.00000003              |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | LSTM                               | 40             | 18                  | 2                         | 0.00000889               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Selective SSM (Mamba-style)        | 40             | 21                  | 1                         | 0.00000472               |
| Tỷ suất sinh lợi tích lũy 5 phiên        | 5       | Ridge (alpha chọn bằng validation) | 8              | 6                   | 1                         | 0.00002182               |
| Độ biến động thực hiện 5 phiên tương lai | 5       | LSTM                               | 40             | 16                  | 0                         | -0.00000223              |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Selective SSM (Mamba-style)        | 40             | 17                  | 0                         | -0.00000113              |
| Độ biến động thực hiện 5 phiên tương lai | 5       | Ridge (alpha chọn bằng validation) | 8              | 2                   | 0                         | -0.00000486              |

Bảng này ghép đúng cùng mô hình, mã và seed. Cải thiện MSE dương nghĩa là Market+News tốt hơn Market-only. Các seed trên cùng mã dùng chung chuỗi test; p-value HAC là bằng chứng bổ sung cho từng cặp, chưa điều chỉnh nhiều lần.

## Giới hạn khi diễn giải

PhoBERT được huấn luyện trên phản hồi khách hàng ngân hàng. Cần nhãn người cho mẫu tiêu đề chứng khoán để đánh giá đúng miền đích. Metadata RSS được truy hồi hồi cứu; thời điểm xuất bản không xác nhận thời điểm bản tin đã có sẵn trong chỉ mục lịch sử. Vì vậy kết quả tin tức chưa phải kiểm định giao dịch point-in-time.

Manifest lần chạy: Git `f621a3467390620fb076934c7de003e88fbdd8b1`, Python 3.13.13.

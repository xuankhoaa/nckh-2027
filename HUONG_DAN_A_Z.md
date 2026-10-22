# Hướng dẫn A–Z: dự báo tỷ suất sinh lợi cổ phiếu ngân hàng bằng Mamba

Tài liệu này mô tả chính xác những gì đã được tạo, dữ liệu đi vào đâu, kết quả đi ra đâu và cách lặp lại thí nghiệm.

## A. Mục tiêu nghiên cứu

Dự báo **log return của phiên giao dịch t+1** từ thông tin có đến cuối phiên **t**. Đây không phải là bài toán dự đoán raw closing price.

Ba mô hình được so sánh:

1. `lstm`: baseline tuần tự.
2. `mamba_baseline`: lõi MambaStock, feature bắt nguồn từ OHLCV.
3. `improved_mamba`: cùng lõi và thiết kế với baseline, thêm return của VN-Index.

## B. Môi trường đã dùng

- Windows, Python 3.13.
- Virtual environment dùng chung: `C:\Users\nguye\.venv`.
- Các gói: `vnstock`, `vnai`, `python-dotenv`, `pandas`, `scikit-learn`, `matplotlib`, `torch`.
- Vnstock đọc API key từ `.env`; khóa không được ghi vào mã nguồn hay bảng kết quả.

## C. Cấu trúc thư mục

```text
NCKH/
├── data/                         # CSV OHLCV cache từ Vnstock
├── external/MambaStock/          # repository nền, giữ nguyên mamba.py và pscan.py
├── models/
│   ├── lstm.py                   # LSTM baseline
│   └── mamba_model.py            # regression head gọi MambaStock gốc
├── results/                      # bảng và biểu đồ sinh ra sau mỗi lần chạy
├── get_data.py                   # lấy/cache dữ liệu
├── preprocess.py                 # feature, split và sliding window
├── train.py                      # chạy một model/một mã
├── run_experiments.py            # chạy 3 model cho toàn bộ mã
├── generate_report.py            # sinh báo cáo từ metrics.csv
├── config.py                     # danh sách ticker và cấu hình split
├── .env                          # API key thực, không commit
├── .env.example                  # mẫu biến môi trường
└── requirements.txt              # dependency
```

## D. Đầu vào dữ liệu

`get_data.py` lấy dữ liệu daily từ Vnstock nguồn KBS cho:

- Cổ phiếu: `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB`.
- Chỉ số thị trường: `VNINDEX`.

Schema CSV được kiểm tra trước khi ghi:

| Cột | Ý nghĩa |
| --- | --- |
| `time` | Ngày giao dịch, chuẩn `YYYY-MM-DD` |
| `open`, `high`, `low`, `close` | OHLC giá ngày |
| `volume` | Khối lượng giao dịch |

CSV đã có sẽ được dùng lại. Chỉ dùng `--force` khi chủ ý cập nhật nguồn dữ liệu.

## E. Thiết lập API key an toàn

Tạo `.env` cạnh `get_data.py` theo `.env.example`:

```text
VNSTOCK_API_KEY=your_api_key_here
```

`.gitignore` đã loại trừ `.env`. Không dán key vào code, notebook hoặc báo cáo.

## F. Feature engineering

Từ dữ liệu mỗi cổ phiếu:

```text
return_t        = log(close_t / close_(t-1))
volume_change_t = volume_t / volume_(t-1) - 1
high_low_range  = (high_t - low_t) / close_t
```

Từ VN-Index:

```text
vnindex_return_t = log(vnindex_close_t / vnindex_close_(t-1))
```

Feature baseline gồm OHLCV và ba đại lượng dẫn xuất ở trên. Improved Mamba thêm `vnindex_return`.

## G. Target và sliding window

Target là:

```text
target_return_t = return_(t+1)
```

Với `window=30`, một sample có dạng:

```text
X = [feature_(t-29), ..., feature_t]
y = return_(t+1)
```

Tensor đi vào model có shape `[batch_size, 30, num_features]`; baseline có 8 feature, Improved Mamba có 9 feature. Chương trình in một sample alignment và assert `input_end_date < target_date` trước khi train.

## H. Kiểm soát data leakage

1. `target_return` và `target_date` được tạo bằng `shift(-1)`.
2. Một sample được phân vào train/validation/test theo **ngày target**, không theo thứ tự ngẫu nhiên.
3. `StandardScaler.fit()` chỉ dùng quan sát đến ngày input cuối cùng thuộc train.
4. Validation và test không tham gia fit scaler, chọn epoch hay cập nhật trọng số.
5. Cửa sổ test được phép nhìn về các ngày quá khứ, nhưng không nhìn bất kỳ feature nào sau ngày target.

## I. Chia train/validation/test

| Tập | Khoảng target date | Vai trò |
| --- | --- | --- |
| Train | đến 31/12/2023 | học trọng số và scaler |
| Validation | năm 2024 | chọn epoch tốt nhất/early stopping |
| Test | từ 01/01/2025 | chỉ đo kết quả cuối cùng |

## J. MambaStock được tái sử dụng thế nào

Repository nền nằm ở `external/MambaStock/`.

- Giữ nguyên `mamba.py`: selective state-space blocks, residual blocks, RMSNorm.
- Giữ nguyên `pscan.py`: selective scan song song.
- Không dùng trực tiếp `main.py`: bản gốc gắn target `pct_chg` cho cùng hàng, không dùng fixed sliding windows, không có validation và quy đổi kết quả về close price. Các đặc điểm đó không khớp câu hỏi t → t+1.

`models/mamba_model.py` chỉ thêm input projection và regression head lấy hidden state cuối của một window.

## K. Huấn luyện một thực nghiệm

LSTM cho VCB:

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe train.py --model lstm --ticker VCB --window 30 --epochs 75
```

Improved Mamba cho VCB:

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe train.py --model mamba --ticker VCB --include-vnindex --window 30 --epochs 75
```

`train.py` dùng MSE loss, Adam, seed 42 và early stopping theo validation loss. Test chỉ được dự báo sau khi nạp lại trọng số validation tốt nhất.

## L. Chạy toàn bộ nhóm ngân hàng

```powershell
& C:\Users\nguye\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
& C:\Users\nguye\.venv\Scripts\python.exe run_experiments.py --window 30 --epochs 75 --batch-size 128
& C:\Users\nguye\.venv\Scripts\python.exe generate_report.py --window 30
```

Lệnh thứ hai chạy 8 mã × 3 mô hình. Kết quả cùng cấu hình sẽ được cập nhật, không nhân bản dòng cũ.

## M. Metric đầu ra

| Metric | Cách diễn giải |
| --- | --- |
| RMSE | Sai số bình phương trung bình căn bậc hai; thấp hơn tốt hơn. |
| MAE | Sai số tuyệt đối trung bình; thấp hơn tốt hơn. |
| R² | Mức giải thích biến thiên; cao hơn tốt hơn. |
| Directional Accuracy | Tỷ lệ dự báo cùng dấu với return thực tế; cao hơn tốt hơn. |

## N. Các tệp đầu ra

| Tệp | Nội dung |
| --- | --- |
| `results/BANG_KET_QUA_CHI_TIET_VI.csv` | Bảng metric đầy đủ theo từng mã, cột tiếng Việt, mở đúng dấu trong Excel. |
| `results/BANG_TONG_HOP_NHOM_VI.csv` | Trung bình và độ lệch chuẩn nhóm ngân hàng, cột tiếng Việt. |
| `results/DU_BAO_THEO_NGAY_VI.csv` | Dự báo từng ngày test, cột tiếng Việt. |
| `results/metrics.csv` | Một dòng cho mỗi model × ticker, gồm bốn metric. |
| `results/predictions.csv` | `date`, `ticker`, model, actual return, predicted return. |
| `results/group_summary.csv` | Mean và standard deviation theo mô hình trên 8 mã. |
| `results/BAO_CAO_KET_QUA.md` | Bảng chi tiết và bảng trung bình được tạo tự động. |
| `results/bieu_do/tung_ma/` | Biểu đồ return thật/dự báo, loss và so sánh metric cho từng mã. |
| `results/bieu_do/tong_hop/` | Biểu đồ RMSE, MAE và directional accuracy trung bình toàn nhóm. |

## O. Cách dùng cho báo cáo NCKH

1. Trích xuất bảng trong `BAO_CAO_KET_QUA.md` hoặc `metrics.csv`.
2. Báo cáo mean ± standard deviation ở `group_summary.csv` thay vì chỉ chọn một mã đẹp nhất.
3. So sánh Mamba baseline với Improved Mamba cho cùng ticker trước, sau đó so sánh trung bình nhóm.
4. Khi metric xấu hoặc R² âm, báo cáo trung thực và thảo luận rằng dự báo return ngắn hạn rất nhiễu; không sửa số liệu.
5. Thử window 60/90 hoặc nhiều seed như thực nghiệm tiếp theo, lập kế hoạch trước và không chọn theo test set.

## P. Các giới hạn hiện tại

- Đây là dự báo one-day-ahead return, không phải khuyến nghị giao dịch.
- Chưa có transaction cost, backtest, tin tức hay sentiment.
- Chỉ có một seed cho cấu hình chính; cần nhiều seed trước khi kết luận mạnh.
- Dữ liệu phụ thuộc vào nguồn Vnstock và các ngày giao nhau với VN-Index.

## Q. Mở rộng hợp lý tiếp theo

Sau khi chốt bảng 30 ngày, chạy lại toàn bộ với `--window 60` và `--window 90`, rồi thêm cột window vào phân tích. Chỉ nghiên cứu RSS/news sau khi các thí nghiệm lõi đã ổn định.

## R. Kiểm tra nhanh khi có lỗi

- Thiếu key: xem `.env.example`, đảm bảo `.env` có `VNSTOCK_API_KEY`.
- Thiếu CSV: chạy lại `get_data.py` cho mã cần dùng.
- Muốn cập nhật nguồn: thêm `--force`.
- Thiếu package: dùng đúng Python trong `C:\Users\nguye\.venv`.
- Không khớp dữ liệu: `preprocess.py` sẽ báo lỗi thiếu cột hoặc không có sequence thay vì âm thầm tiếp tục.

## S. Tính tái lập

Seed mặc định là 42. Mỗi dòng metrics mang ticker, model, window và feature set. Giữ nguyên CSV cache cùng lệnh chạy để tái lập kết quả.

## T. Điều không được làm

- Không hard-code API key.
- Không fit scaler trên toàn bộ dữ liệu.
- Không shuffle time split.
- Không dùng return t+1 trong input.
- Không thay kết quả thật bằng số liệu minh họa.

## U. Kết luận vận hành

Toàn bộ luồng là: **Vnstock CSV → feature tại t → sliding window → LSTM/Mamba → return t+1 → CSV/biểu đồ/báo cáo**.

Các chữ cái còn lại V–Z không biểu thị bước kỹ thuật mới; chúng nhấn mạnh yêu cầu cuối: kiểm tra kết quả, xác minh leakage, lưu output, và trình bày trung thực trong báo cáo.

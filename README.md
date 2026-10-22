# NCKH 2027 — Dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Selective SSM

## Mục tiêu và phạm vi

Kho mã này phục vụ **Bài toán 1 — dự báo tỷ suất sinh lợi cổ phiếu**: từ thông tin có đến cuối phiên giao dịch $t$, dự báo **log-return** của phiên $t+1$ cho tám cổ phiếu ngân hàng Việt Nam. Đây là bài toán hồi quy, không phải dự báo giá đóng cửa và không phải khuyến nghị đầu tư.

$$
y_t = r_{t+1}=\ln\left(\frac{P_{t+1}}{P_t}\right).
$$

Phạm vi hiện tại chỉ giải quyết Bài toán 1. Kho mã chưa tạo target độ biến động tương lai (Bài toán 2) và chưa gán nhãn rủi ro thấp/trung bình/cao để làm phân lớp (Bài toán 3). Hai hướng đó được nêu như mở rộng, không được trình bày như kết quả đã đạt được.

Tài liệu nền tảng:

- [Công thức, thuật toán và đề cương NCKH](CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md)
- [Hướng dẫn vận hành A–Z](HUONG_DAN_A_Z.md)
- [Ghi chú nguồn tham khảo](THIRD_PARTY_NOTICES.md)

## Dữ liệu đầu vào

| Thành phần | Nội dung |
|---|---|
| Mã cổ phiếu | `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB` |
| Chỉ số thị trường | `VNINDEX` |
| Nguồn | OHLCV ngày từ Vnstock/KBS, lưu cache CSV cục bộ |
| Snapshot thiết kế | 01/01/2020–12/10/2026; ngày thực tế của từng tệp được ghi trong manifest |
| Cửa sổ đầu vào | 30 phiên giao dịch |
| Đầu ra mỗi mẫu | Một log-return liên tục của phiên kế tiếp |

Mỗi CSV phải có sáu cột: `time`, `open`, `high`, `low`, `close`, `volume`. `data_quality.py` kiểm tra cột bắt buộc, trùng ngày, giá trị thiếu/không hữu hạn, điều kiện OHLC dương, `high >= low`, khối lượng không âm và phạm vi `close`. Mã SHA-256, số dòng, ngày đầu/cuối và cảnh báo được lưu vào `data/DATA_MANIFEST.json` khi chạy pipeline. Dữ liệu gốc không bị tự sửa.

## Đặc trưng, nhãn và chống rò rỉ dữ liệu

Từ mỗi phiên $t$, chương trình tạo:

| Đặc trưng | Công thức |
|---|---|
| `return` | $\ln(P_t/P_{t-1})$ |
| `volume_change` | $V_t/V_{t-1}-1$ |
| `high_low_range` | $(H_t-L_t)/P_t$ |
| `vnindex_return` | $\ln(I_t/I_{t-1})$ |

LSTM và Selective SSM cơ sở dùng tám biến: OHLCV cùng ba đặc trưng dẫn xuất đầu. Selective SSM có VN-Index dùng thêm `vnindex_return`, nên có chín biến. Mỗi mẫu có dạng $X_t=[x_{t-29},\ldots,x_t]$ và nhãn $y_t=r_{t+1}$.

Mẫu được gán theo **ngày của nhãn**, không chia ngẫu nhiên:

| Tập | Ngày của nhãn | Vai trò |
|---|---|---|
| Train | Đến 31/12/2023 | Fit scaler, target scaler và trọng số |
| Validation | 01/01/2024–31/12/2024 | Chọn epoch dừng sớm |
| Test | Từ 01/01/2025 | Chỉ đánh giá cuối cùng |

`StandardScaler` chỉ fit trên feature của train. Với mạng nơ-ron, target cũng được chuẩn hóa bằng trung bình và độ lệch chuẩn **của train** để tối ưu ổn định, rồi đảo chuẩn hóa trước khi tính metric. Cửa sổ test được phép chứa lịch sử trước đó nhưng không thể chứa feature hoặc nhãn sau ngày cần dự báo.

## Mô hình được đánh giá

| ID | Mô hình | Vai trò |
|---|---|---|
| `naive_zero` | Luôn dự báo return bằng 0 | Baseline “không thay đổi” |
| `train_mean` | Luôn dự báo trung bình return của train | Baseline hằng, chỉ dùng train |
| `lstm` | LSTM một lớp, hidden size 8, 585 tham số (8 đầu vào) | Đối chứng tuần tự |
| `mamba_baseline` | Selective SSM một lớp, $d_{state}=8$, causal convolution kernel 3, 545 tham số | Đánh giá SSM với OHLCV dẫn xuất |
| `improved_mamba` | Selective SSM như trên, thêm return VN-Index, 553 tham số | Kiểm tra đóng góp thị trường chung |

`models/mamba_model.py` là hiện thực Selective SSM tự viết từ mô tả trong bài báo Mamba. Nó dùng RMSNorm, chiếu đầu vào, causal depthwise convolution, tham số chọn lọc $\Delta,B,C$, kết nối residual và cổng SiLU. Phép quét trạng thái dùng phép ghép affine kết hợp theo cây; test đơn vị xác nhận nó tương đương truy hồi tuần tự. Kho không chứa mã nguồn vendored từ MambaStock.

## Thiết kế thực nghiệm và đánh giá

Mỗi mạng được lặp lại trên năm seed: `42, 52, 62, 72, 82`. Mỗi lần chạy dùng Adam (`learning_rate=0.001`), tối đa 40 epoch, batch size 128, early stopping với patience 8 theo MSE validation. Hai baseline là xác định nên chỉ cần seed 0.

Lệnh chạy đầy đủ tạo đúng **136 lần chạy**:

$$
8\ \text{mã}\times\left(2\ \text{baseline}+3\ \text{mạng}\times5\ \text{seed}\right)=136.
$$

Các metric trên test gồm RMSE, MAE, $R^2$ và Directional Accuracy (DA). Báo cáo còn có kiểm định chênh lệch squared loss với phương sai HAC khi so sánh mỗi seed mạng với baseline return bằng 0. Không chọn mô hình chỉ theo DA; RMSE/MAE đo độ lớn sai số. $R^2$ âm phải được báo cáo trung thực, vì nó cho thấy mô hình kém hơn dự báo hằng bằng trung bình tập test theo thước đo đó.

## Kết quả của lượt chạy đã kiểm chứng

Lượt chạy đầy đủ với snapshot dữ liệu đến 12/10/2026 đã hoàn thành đủ 136 cấu hình. Bảng sau là trung bình nhóm trên tám mã; với RMSE và MAE, **nhỏ hơn là tốt hơn**.

| Mô hình | RMSE | MAE | $R^2$ | DA |
|---|---:|---:|---:|---:|
| Baseline return bằng 0 | 0,018928 | 0,012859 | -0,001364 | 0,078588 |
| Baseline trung bình train | **0,018918** | 0,012899 | -0,000482 | 0,462415 |
| LSTM | 0,019195 | 0,013171 | -0,030056 | 0,459055 |
| Selective SSM cơ sở | 0,019145 | 0,013180 | -0,024805 | 0,459510 |
| Selective SSM có VN-Index | 0,019156 | 0,013211 | -0,026057 | 0,461048 |

Kết luận đúng với cấu hình hiện tại là: baseline trung bình train có RMSE thấp nhất; mạng tốt nhất là Selective SSM cơ sở nhưng chưa vượt baseline. Đây là kết quả âm có giá trị khoa học, không phải lỗi bị che giấu. Bảng theo từng mã, độ lệch chuẩn seed, kiểm định HAC và diễn giải đầy đủ nằm trong [báo cáo NCKH](bao_cao/BAO_CAO_NCKH.md).

## Cài đặt và chạy lại

`get_data.py` dùng API `Quote` thuộc dòng **Vnstock 4.x**, vì vậy `requirements.txt` khóa `vnstock<5`. Không cài Vnstock 5.x chồng lên cùng môi trường. Để tải dữ liệu mới, hãy tạo môi trường riêng và thiết lập khóa API của bạn trong `.env`; tuyệt đối không đưa khóa vào Git.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
Copy-Item .env.example .env
```

`VNSTOCK_API_KEY` chỉ cần khi tải mới hoặc làm mới CSV. Nếu sử dụng snapshot CSV đã cache trong `data/`, có thể bỏ qua `get_data.py` và không cần gọi API. Khi cần tải dữ liệu, điền khóa vào `.env` rồi chạy:

```powershell
.\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Chạy lại thí nghiệm từ snapshot dữ liệu hiện có:

```powershell
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe generate_report.py --window 30
.\.venv\Scripts\python.exe verify_project.py --require-results
```

Mặc định `get_data.py` tái sử dụng CSV đã cache. Chỉ thêm `--force` khi chủ ý ghi đè snapshot dữ liệu. `run_experiments.py` tự xóa đúng các output tái tạo được trước một lượt chạy mới; dùng `--keep-results` chỉ khi hiểu rõ mình đang tiếp tục tập kết quả cùng schema.

Trước khi nộp hoặc sau khi chỉnh mã, kiểm tra nhanh bằng:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe verify_project.py --require-results
```

## Đầu ra

Tất cả kết quả sinh tự động ở `results/` (được Git bỏ qua vì tái tạo được):

| Tệp | Nội dung |
|---|---|
| `EXPERIMENT_MANIFEST.json` | Commit, phiên bản môi trường, thiết bị, cấu hình và manifest dữ liệu |
| `BANG_KET_QUA_CHI_TIET_VI.csv` | 136 metric theo mã, mô hình và seed; tiêu đề tiếng Việt |
| `BANG_TONG_HOP_NHOM_VI.csv` | Trung bình và độ lệch chuẩn giữa mã/seed |
| `DU_BAO_THEO_NGAY_VI.csv` | Log-return thực tế và dự báo từng ngày test; tiêu đề tiếng Việt |
| `KIEM_DINH_HAC_VI.csv` | Kiểm định chênh lệch squared loss so với baseline return bằng 0 |
| `BAO_CAO_KET_QUA.md` | Báo cáo tóm tắt sinh tự động từ chính các CSV |
| `bieu_do/` | Learning curve, thực tế–dự báo và biểu đồ metric nhóm |

`generate_report.py` cũng tạo [bản báo cáo Markdown đầy đủ](bao_cao/BAO_CAO_NCKH.md) để làm nguồn cho người phụ trách soạn Word. Báo cáo không thay Word; nó là nội dung có thể kiểm tra bằng mã và kết quả hiện hành.

## Cấu trúc mã nguồn

| Tệp | Vai trò |
|---|---|
| `config.py` | Snapshot dữ liệu, ticker, split, window và seed |
| `get_data.py` | Tải/cache OHLCV và tạo manifest dữ liệu |
| `data_quality.py` | Xác thực OHLCV và SHA-256 dữ liệu |
| `preprocess.py` | Feature, target, scaler và cửa sổ không rò rỉ |
| `baselines.py` | Hai baseline hằng |
| `models/lstm.py` | LSTM đối chứng |
| `models/mamba_model.py` | Selective SSM tự triển khai |
| `train.py` | Huấn luyện một cấu hình và lưu dự báo/metric |
| `run_experiments.py` | Chạy đầy đủ 136 cấu hình |
| `evaluate.py` | Metric, CSV và biểu đồ |
| `generate_report.py` | Báo cáo/bảng tiếng Việt từ kết quả |
| `console.py` | Chuẩn hóa đầu ra UTF-8 cho các lệnh chính trên Windows |
| `verify_project.py` | Xác minh mã, dữ liệu cache và output |
| `tests/` | Test chất lượng dữ liệu, leakage, scan và metric |

## Giới hạn và hướng mở rộng

- Đây là dự báo one-day-ahead return với feature giá/khối lượng và VN-Index; chưa dùng tin tức, sentiment, yếu tố vĩ mô hoặc báo cáo tài chính.
- Chưa có transaction cost, chiến lược giao dịch hay backtest lợi nhuận. Không dùng output làm khuyến nghị mua/bán.
- Để giải quyết Bài toán 2, cần định nghĩa target độ biến động tương lai (ví dụ tổng bình phương return trong $h$ phiên tới) và đánh giá RMSE/MAE riêng.
- Để giải quyết Bài toán 3, cần quy tắc gán nhãn rủi ro từ target tương lai, xử lý mất cân bằng lớp và dùng metric classification như macro-F1, balanced accuracy, confusion matrix.
- Bước nghiên cứu tiếp theo hợp lý là walk-forward validation và kiểm tra độ bền qua window 60/90, được lập kế hoạch trước thay vì chọn theo test.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
5. Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610

# NCKH 2027 — Dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Selective SSM

Kho mã nguồn này là pipeline nghiên cứu có thể tái lập để dự báo **log-return một phiên kế tiếp** của tám cổ phiếu ngân hàng Việt Nam. Dự án được xây dựng cho mục tiêu học thuật: kiểm tra liệu Selective State Space Model (Selective SSM) lấy cảm hứng từ Mamba có giảm sai số dự báo so với các baseline đơn giản và LSTM hay không.

> Đây không phải công cụ tư vấn đầu tư, không dự báo giá đóng cửa và không tạo tín hiệu mua/bán.

## Tổng quan

| Hạng mục | Nội dung |
|---|---|
| Bài toán | Hồi quy tỷ suất sinh lợi (Bài toán 1) |
| Mục tiêu | Dự báo $r_{t+1}=\ln(P_{t+1}/P_t)$ từ thông tin có đến cuối phiên $t$ |
| Mã nghiên cứu | `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB` |
| Dữ liệu | OHLCV ngày và return VN-Index, nguồn Vnstock/KBS |
| Khoảng snapshot | 01/01/2020–12/10/2026; ngày giao dịch đầu tiên thực tế là 02/01/2020 |
| Cửa sổ đầu vào | 30 phiên giao dịch |
| Mô hình | 2 baseline, LSTM, Selective SSM cơ sở, Selective SSM có VN-Index |
| Đánh giá | RMSE, MAE, $R^2$, Directional Accuracy và kiểm định HAC |

Các bài toán dự báo độ biến động tương lai và phân loại rủi ro thấp/trung bình/cao **không nằm trong phạm vi hiện tại**. Chúng cần biến mục tiêu, thiết kế đánh giá và metric riêng; dự án không trình bày chúng như kết quả đã đạt được.

## Câu hỏi nghiên cứu

1. Trên cùng dữ liệu, cửa sổ và giai đoạn kiểm tra, Selective SSM có dự báo log-return một phiên kế tiếp tốt hơn baseline và LSTM không?
2. Bổ sung return VN-Index có giúp Selective SSM giảm sai số dự báo không?

Các câu trả lời chỉ có giá trị trong phạm vi snapshot dữ liệu, feature, horizon và cấu hình đã công bố trong kho này.

## Thiết kế nghiên cứu

### Dữ liệu, feature và nhãn

Mỗi phiên giao dịch cung cấp năm biến OHLCV: `open`, `high`, `low`, `close`, `volume`. Pipeline dẫn xuất ba feature của từng cổ phiếu là `return`, `volume_change`, `high_low_range`; biến thể thị trường chung bổ sung `vnindex_return`.

Với giá đóng cửa $P_t$, khối lượng $V_t$, giá cao/thấp $H_t,L_t$ và giá đóng cửa VN-Index $I_t$:

$$
\begin{aligned}
r_t &= \ln(P_t/P_{t-1}), \\
v_t &= V_t/V_{t-1}-1, \\
q_t &= (H_t-L_t)/P_t, \\
r_t^{VN} &= \ln(I_t/I_{t-1}), \\
y_t &= r_{t+1}.
\end{aligned}
$$

Mỗi mẫu là cửa sổ $X_t=[x_{t-29},\ldots,x_t]$; nhãn là log-return của phiên $t+1$. Vì vậy, feature của tương lai không thể xuất hiện trong đầu vào.

### Chia dữ liệu và chống rò rỉ

Mẫu được gán tập theo **ngày của nhãn**, không chia ngẫu nhiên.

| Tập | Ngày của nhãn | Mục đích |
|---|---|---|
| Train | Đến 31/12/2023 | Fit trọng số, feature scaler và target scaler |
| Validation | 01/01/2024–31/12/2024 | Chọn epoch dừng sớm |
| Test | Từ 01/01/2025 | Đánh giá cuối cùng |

`StandardScaler` chỉ được fit trên feature của train. Với mạng nơ-ron, target được chuẩn hóa bằng thống kê train khi tối ưu và được đảo chuẩn hóa trước khi tính metric. Cửa sổ test có thể dùng lịch sử quá khứ, nhưng không dùng feature hoặc nhãn sau ngày dự báo.

### Mô hình so sánh

| ID | Mô hình | Thiết lập chính | Vai trò |
|---|---|---|---|
| `naive_zero` | Baseline return bằng 0 | 0 tham số | Mốc “không thay đổi” |
| `train_mean` | Baseline trung bình tập huấn luyện | 0 tham số | Mốc hằng chỉ dùng train |
| `lstm` | LSTM | 1 lớp, hidden size 8, 585 tham số | Đối chứng mạng tuần tự |
| `mamba_baseline` | Selective SSM cơ sở | 1 lớp, $d_{state}=8$, causal convolution kernel 3, 545 tham số | Đánh giá Selective SSM trên feature cổ phiếu |
| `improved_mamba` | Selective SSM có VN-Index | Như trên, 553 tham số | Kiểm tra giá trị tăng thêm của thị trường chung |

`models/mamba_model.py` là hiện thực Selective SSM độc lập, tối giản theo ý tưởng Mamba: RMSNorm, chiếu đầu vào, causal depthwise convolution, tham số chọn lọc $\Delta,B,C$, cổng SiLU, residual connection và prefix scan affine. Test đơn vị xác nhận phép quét song song khớp truy hồi tuần tự.

MambaStock chỉ được sử dụng làm tài liệu tham khảo học thuật. Kho này không fork, import hoặc chỉnh sửa tiếp mã nguồn MambaStock; vì vậy không nên gọi kết quả là bản tái lập nguyên xi MambaStock hoặc benchmark của thư viện `mamba_ssm` chính thức.

### Thiết lập thực nghiệm

Mỗi mạng được chạy với năm seed `42, 52, 62, 72, 82`; hai baseline có tính xác định nên chỉ chạy seed `0`. Adam dùng learning rate `0.001`, tối đa 40 epoch, batch size 128 và early stopping patience 8 theo MSE validation.

Tổng số lượt chạy của bộ thực nghiệm đầy đủ là:

$$
8\ \text{mã}\times\left(2\ \text{baseline}+3\ \text{mạng}\times5\ \text{seed}\right)=136.
$$

## Kết quả thực nghiệm

Lượt chạy đầy đủ trên snapshot đến 12/10/2026 đã hoàn thành 136 cấu hình. Bảng dưới là trung bình nhóm trên tám mã; RMSE và MAE càng nhỏ càng tốt.

| Mô hình | RMSE | MAE | $R^2$ | DA |
|---|---:|---:|---:|---:|
| Baseline return bằng 0 | 0,018928 | 0,012859 | -0,001364 | 0,078588 |
| Baseline trung bình tập huấn luyện | **0,018918** | 0,012899 | -0,000482 | 0,462415 |
| LSTM | 0,019195 | 0,013171 | -0,030056 | 0,459055 |
| Selective SSM cơ sở | 0,019145 | 0,013180 | -0,024805 | 0,459510 |
| Selective SSM có VN-Index | 0,019156 | 0,013211 | -0,026057 | 0,461048 |

Kết quả cần được diễn giải thận trọng: baseline trung bình tập huấn luyện có RMSE thấp nhất. Trong các mạng, Selective SSM cơ sở tốt nhất, nhưng RMSE vẫn cao hơn baseline `0,000227`. Do đó, nghiên cứu **chưa có bằng chứng** rằng Selective SSM giảm sai số trung bình nhóm ở cấu hình hiện tại; thêm return VN-Index cũng chưa cải thiện RMSE. Đây là kết quả âm được giữ lại và báo cáo minh bạch.

Phân tích theo từng mã, độ lệch chuẩn theo seed, kiểm định HAC, bảng dự báo theo ngày và diễn giải học thuật có trong [báo cáo NCKH đầy đủ](bao_cao/BAO_CAO_NCKH.md).

## Tái lập kết quả

### 1. Chuẩn bị môi trường

Yêu cầu Python 3.10 trở lên. Lệnh kiểm tra trong kho đã được chạy với Python 3.13.

```powershell
git clone https://github.com/xuankhoaa/nckh-2027.git
cd nckh-2027
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
```

`get_data.py` dùng API `Quote` của dòng Vnstock 4.x, và `requirements.txt` đã khóa `vnstock<5`. Hãy giữ môi trường `.venv` này riêng cho dự án; không nâng cấp chồng Vnstock 5.x vào môi trường đó.

### 2. Chuẩn bị dữ liệu

Thư mục `data/` được Git bỏ qua vì chứa snapshot tái tạo được. Nếu checkout của bạn đã có các CSV cache đúng snapshot, có thể chuyển thẳng sang bước 3. Nếu chưa có, tạo tệp `.env` từ mẫu, điền khóa API của bạn tại máy cục bộ và tải dữ liệu:

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Không đưa `VNSTOCK_API_KEY` vào mã nguồn, notebook, commit hoặc cuộc trò chuyện. `get_data.py` luôn bổ sung `VNINDEX`, tái sử dụng CSV đã cache theo mặc định và chỉ ghi đè khi có cờ `--force`.

Sau bước dữ liệu, `data/DATA_MANIFEST.json` lưu số dòng, khoảng ngày, SHA-256 và cảnh báo chất lượng. Snapshot hiện hành có 9 tệp OHLCV; có cảnh báo nguồn cho 3 dòng VN-Index có `close` ngoài khoảng `[low, high]`. Pipeline chỉ dùng `close` để tính `vnindex_return` và không tự sửa dữ liệu nguồn.

### 3. Chạy thí nghiệm và sinh báo cáo

```powershell
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe generate_report.py --window 30
.\.venv\Scripts\python.exe verify_project.py --require-results
```

`run_experiments.py` xóa đúng các output tái tạo được trước một lượt chạy mới. Chỉ dùng `--keep-results` khi chủ ý tiếp tục một bộ kết quả đang có cùng schema.

### 4. Kiểm tra trước khi nộp

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe verify_project.py --require-results
```

`verify_project.py` kiểm tra sự hiện diện của mã cốt lõi, tính hợp lệ của dữ liệu cache, tính đầy đủ của 136 dòng metric và sự khớp giữa metric với dự báo.

## Đầu ra và tính tái lập

Kết quả tái tạo được sinh trong `results/` và không được đưa vào Git. Những tệp chính gồm:

| Tệp | Nội dung |
|---|---|
| `EXPERIMENT_MANIFEST.json` | Commit Git, phiên bản môi trường, thiết bị, cấu hình và manifest dữ liệu |
| `BANG_KET_QUA_CHI_TIET_VI.csv` | 136 metric theo mã, mô hình và seed; cột tiếng Việt |
| `BANG_TONG_HOP_NHOM_VI.csv` | Trung bình/độ lệch chuẩn theo mã và seed |
| `DU_BAO_THEO_NGAY_VI.csv` | Giá trị thực và dự báo log-return theo ngày test |
| `KIEM_DINH_HAC_VI.csv` | So sánh squared loss của mạng với baseline return bằng 0 |
| `BAO_CAO_KET_QUA.md` | Tóm tắt tự động từ các CSV |
| `bieu_do/` | Learning curve, thực tế–dự báo và biểu đồ metric nhóm |

Manifest giữ mã SHA-256 của dữ liệu cùng commit mã nguồn, nhờ đó một lượt chạy có thể xác định được chính xác đầu vào và môi trường đã dùng.

## Cấu trúc kho mã

```text
.
├── config.py                 # Snapshot dữ liệu, ticker, split, window, seed
├── get_data.py               # Tải/cache OHLCV và lập manifest dữ liệu
├── data_quality.py           # Xác thực schema, OHLCV và SHA-256
├── preprocess.py             # Feature, target, scaler và cửa sổ thời gian
├── baselines.py              # Hai baseline hằng
├── train.py                  # Huấn luyện một cấu hình
├── run_experiments.py        # Điều phối toàn bộ 136 lượt chạy
├── evaluate.py               # Metric, CSV và biểu đồ
├── generate_report.py        # Sinh báo cáo/bảng tiếng Việt
├── provenance.py             # Dấu vết môi trường và cấu hình thực nghiệm
├── verify_project.py         # Xác minh tính toàn vẹn của dự án
├── models/                   # LSTM và Selective SSM
├── tests/                    # Test dữ liệu, leakage, scan và metric
└── bao_cao/                  # Báo cáo Markdown dùng để soạn báo cáo chính thức
```

## Giới hạn

- Dự án chỉ sử dụng OHLCV và return VN-Index; chưa sử dụng tin tức, sentiment, yếu tố vĩ mô hoặc dữ liệu cơ bản doanh nghiệp.
- Dự báo là one-day-ahead return, chưa có transaction cost, chiến lược giao dịch hoặc backtest lợi nhuận.
- Test hiện là một giai đoạn hậu 01/01/2025; kết quả chưa thay thế walk-forward validation nhiều cửa sổ.
- Bài toán biến động và phân loại rủi ro cần được thiết kế thành các nghiên cứu tiếp theo, không suy ra trực tiếp từ kết quả hồi quy return này.

## Tài liệu dự án

- [Báo cáo NCKH đầy đủ](bao_cao/BAO_CAO_NCKH.md): diễn giải phương pháp, kết quả, kiểm định và giới hạn.
- [Hướng dẫn vận hành A–Z](HUONG_DAN_A_Z.md): đầu vào, đầu ra và quy trình chạy chi tiết.
- [Công thức và thuật toán](CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md): ký hiệu, công thức, Selective SSM và metric.
- [Ghi chú bên thứ ba](THIRD_PARTY_NOTICES.md): phạm vi sử dụng các tài liệu/mã tham khảo.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8
4. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
5. Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610

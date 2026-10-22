# Hướng dẫn A–Z: chạy và kiểm tra đề tài NCKH

Tài liệu này mô tả đầy đủ luồng dữ liệu, cách chạy lại, cách kiểm tra kết quả và ý nghĩa các đầu ra. Bản hiện tại giải quyết **dự báo log-return một phiên kế tiếp** (Bài toán 1), không giải quyết trực tiếp dự báo biến động hay phân loại rủi ro.

## A. Luồng tổng quát

```text
OHLCV cổ phiếu + OHLCV VN-Index
        ↓
Kiểm tra chất lượng + đóng băng manifest
        ↓
Feature tại t + target return tại t+1
        ↓
Chia train / validation / test theo thời gian
        ↓
2 baseline + LSTM + 2 Selective SSM, lặp 5 seed
        ↓
CSV tiếng Việt + biểu đồ + báo cáo Markdown + manifest tái lập
```

## B. Điều kiện môi trường

Mã tải dữ liệu đang dùng `Quote` của **Vnstock 4.x**. Vì vậy, phải dùng một môi trường tách riêng cho dòng 4.x; không nâng cấp chồng Vnstock 5.x vào cùng môi trường.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env` và điền khóa của bạn vào `VNSTOCK_API_KEY`. Tệp `.env` đã bị Git bỏ qua. Không gửi khóa qua chat, không đưa vào notebook, mã nguồn, kết quả hay commit.

Nếu chỉ chạy mô hình từ các CSV đã cache trong `data/`, bước tải dữ liệu không gọi mạng. Tuy nhiên, muốn **tải lại dữ liệu** thì vẫn cần Vnstock 4.x và khóa hợp lệ.

## C. Đầu vào gốc

`get_data.py` tải OHLCV ngày cho tám mã `VCB BID CTG MBB TCB ACB VPB STB` và luôn có `VNINDEX`. Mốc mặc định được khóa tại 01/01/2020–12/10/2026 để một lần chạy lại không âm thầm kéo dữ liệu mới hơn.

| Cột CSV | Ý nghĩa |
|---|---|
| `time` | Ngày giao dịch, `YYYY-MM-DD` |
| `open`, `high`, `low`, `close` | Giá mở, cao, thấp, đóng cửa |
| `volume` | Khối lượng giao dịch |

Chạy tải/cache:

```powershell
.\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Mặc định một CSV đã tồn tại được dùng lại. Chỉ dùng `--force` khi chủ ý gọi nguồn và ghi đè snapshot:

```powershell
.\.venv\Scripts\python.exe get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB --force
```

Sau bước này, `data/DATA_MANIFEST.json` lưu số dòng, ngày đầu/cuối, SHA-256 và cảnh báo dữ liệu. Đó là bằng chứng để người khác xác nhận đang dùng đúng đầu vào, không phải dữ liệu bí mật.

## D. Kiểm tra chất lượng dữ liệu

`data_quality.py` không tự thay đổi số liệu nguồn. Nó chặn các lỗi thiếu cột, trùng ngày, thiếu dữ liệu, số không hữu hạn, giá OHLC không dương, `high < low`, khối lượng âm và `close` nằm ngoài `[low, high]` đối với cổ phiếu.

Với VN-Index, nếu `close` ngoài phạm vi high–low, chương trình ghi **cảnh báo** vào manifest thay vì sửa hoặc xóa dòng. Trong pipeline hiện tại, VN-Index chỉ đóng góp `close` để tính return thị trường; cảnh báo vẫn phải được nêu minh bạch khi viết báo cáo.

## E. Feature engineering

Với giá đóng cửa cổ phiếu $P_t$, khối lượng $V_t$, high $H_t$, low $L_t$ và giá đóng cửa VN-Index $I_t$:

$$
\begin{aligned}
r_t &= \ln(P_t/P_{t-1}),\\
v_t &= V_t/V_{t-1}-1,\\
q_t &= (H_t-L_t)/P_t,\\
r_t^{VN} &= \ln(I_t/I_{t-1}).
\end{aligned}
$$

Tám feature cơ sở là `open high low close volume return volume_change high_low_range`. Bản Selective SSM cải tiến thêm `vnindex_return` thành chín feature. `vnindex_return` là biến động chỉ số thị trường chung, không phải return của mã đang dự báo.

## F. Target và cửa sổ trượt

Mục tiêu là:

$$
y_t=r_{t+1}=\ln(P_{t+1}/P_t).
$$

Với `window=30`, một mẫu là $X_t=[x_{t-29},\ldots,x_t]$ và nhãn là $y_t$. Do đó input kết thúc ở ngày $t$, còn target luôn là phiên sau. Khi chạy một cấu hình, chương trình in ví dụ ghép ngày để kiểm chứng quan hệ này.

## G. Chia dữ liệu và chống leakage

| Tập | Quy tắc theo `target_date` | Được dùng làm gì |
|---|---|---|
| Train | `<= 2023-12-31` | Fit feature scaler, target scaler, trọng số |
| Validation | Năm 2024 | Chọn epoch tốt nhất, early stopping |
| Test | `>= 2025-01-01` | Chỉ đánh giá cuối |

Mẫu không được shuffle. `StandardScaler.fit()` chỉ nhận các feature train. Mạng còn chuẩn hóa target bằng trung bình/độ lệch chuẩn của `y_train`; `y_valid` và `y_test` chỉ áp dụng đúng tham số đó. Dự báo được đảo chuẩn hóa sang log-return trước khi tính RMSE, MAE, $R^2$ và DA.

## H. Mô hình và tính công bằng

| Mô hình | Số tham số | Ý nghĩa |
|---|---:|---|
| `naive_zero` | 0 | Dự báo return bằng 0 |
| `train_mean` | 0 | Dự báo trung bình return train |
| `lstm` | 585 | Một lớp LSTM, hidden size 8 |
| `mamba_baseline` | 545 | Một lớp Selective SSM, chỉ feature cơ sở |
| `improved_mamba` | 553 | Selective SSM, thêm VN-Index |

Mô hình Selective SSM được viết trong `models/mamba_model.py`, không gọi mã vendored từ MambaStock. Nó gồm RMSNorm, causal depthwise convolution, tham số chọn lọc $\Delta,B,C$, cổng SiLU, residual connection và selective scan. Test đơn vị kiểm tra selective scan theo cây cho kết quả tương đương truy hồi trạng thái tuần tự.

## I. Chạy một cấu hình để kiểm tra

Ví dụ LSTM cho VCB:

```powershell
.\.venv\Scripts\python.exe train.py --model lstm --ticker VCB --seed 42 --window 30 --epochs 40
```

Ví dụ Selective SSM có VN-Index cho VCB:

```powershell
.\.venv\Scripts\python.exe train.py --model improved_mamba --ticker VCB --seed 42 --window 30 --epochs 40
```

`train.py` chỉ chạy đúng một cặp model–ticker–seed. Nó dùng Adam, `learning_rate=0.001`, batch size 128, tối đa 40 epoch và early stopping sau 8 epoch validation không cải thiện. Thực nghiệm riêng có thể thay đổi `--epochs`, `--batch-size`, `--window`, nhưng cần ghi rõ đó không còn là cấu hình chính.

## J. Chạy toàn bộ nghiên cứu

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

Lệnh này tự xóa đúng output sinh tự động cũ, sau đó chạy:

$$
8\ \text{mã} \times (2\ \text{baseline}+3\ \text{mạng}\times5\ \text{seed})=136\ \text{lần chạy}.
$$

Năm seed là `42, 52, 62, 72, 82`; baseline hằng dùng seed 0. Dùng `--keep-results` chỉ khi chủ ý tiếp tục đúng schema cũ, vì kết quả thiếu hoặc lẫn cấu hình sẽ làm báo cáo không hợp lệ.

## K. Sinh báo cáo tiếng Việt

Khi 136 lần chạy xong:

```powershell
.\.venv\Scripts\python.exe generate_report.py --window 30
```

Lệnh chỉ chấp nhận đủ 136 dòng metric của cấu hình chính. Nó tạo:

| Đầu ra | Mục đích |
|---|---|
| `results/BANG_KET_QUA_CHI_TIET_VI.csv` | Mọi metric theo mã, mô hình, seed |
| `results/BANG_TONG_HOP_NHOM_VI.csv` | Mean và độ lệch chuẩn giữa mã/seed |
| `results/DU_BAO_THEO_NGAY_VI.csv` | Actual/predicted log-return theo ngày test |
| `results/KIEM_DINH_HAC_VI.csv` | So sánh squared loss với zero return |
| `results/BAO_CAO_KET_QUA.md` | Tóm tắt tự sinh từ kết quả |
| `bao_cao/BAO_CAO_NCKH.md` | Bản Markdown đầy đủ để chuyển thành Word sau này |
| `results/bieu_do/` | Biểu đồ loss, dự báo–thực tế, so sánh nhóm |

CSV tiếng Việt được lưu UTF-8 with BOM (`utf-8-sig`) để mở bằng Excel tiếng Việt mà không lỗi dấu.

## L. Đọc metric đúng cách

| Metric | Cách đọc |
|---|---|
| RMSE | Càng nhỏ càng tốt; nhạy với lỗi lớn |
| MAE | Càng nhỏ càng tốt; sai số tuyệt đối trung bình |
| $R^2$ | Càng lớn càng tốt; âm nghĩa là kém baseline trung bình test theo chỉ số này |
| DA | Tỷ lệ đúng dấu return; phải đọc cùng RMSE/MAE |

Kiểm định HAC được tính trên chênh lệch squared loss giữa một seed mạng và baseline zero ở các ngày test tương ứng. “Cải thiện MSE so với zero” dương nghĩa là mạng giảm loss trung bình. p-value là bằng chứng hỗ trợ, không thay thế việc xem độ lớn sai số, độ bền qua seed và từng mã.

## M. Xác minh trước khi nộp

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe verify_project.py --require-results
```

Test bao phủ kiểm tra OHLCV, target sau input, scaler không bị ảnh hưởng bởi giá trị tương lai, output model, selective scan và metric hoàn hảo. `verify_project.py` kiểm tra layout mã, cache dữ liệu, đủ 136 lần chạy, tính hữu hạn của dự báo/metric và ghép đúng predictions với metrics.

## N. Git và dọn dẹp

`.env`, `data/`, `results/`, checkpoint và Python cache bị Git bỏ qua. Chúng là secret, input thô hoặc output tái tạo được. Mã, test, tài liệu, cấu hình và báo cáo Markdown nguồn được giữ lại. Không chạy lệnh xóa dữ liệu gốc khi chưa sao lưu hoặc chưa xác nhận đúng snapshot cần loại bỏ.

## O. Các lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `No module named vnstock` | Dùng đúng Python trong `.venv`, cài theo `requirements.txt`; giữ Vnstock 4.x |
| Thiếu `VNSTOCK_API_KEY` | Tạo `.env` từ `.env.example`, đặt khóa tại máy cục bộ; không commit |
| Thiếu CSV | Chạy `get_data.py` cho mã cần thiết |
| Báo thiếu 136 dòng metric | Chạy lại `run_experiments.py` không dùng `--keep-results` |
| Lỗi dấu tiếng Việt trong Excel | Mở các tệp hậu tố `_VI.csv`; chúng đã dùng UTF-8 BOM |
| Kết quả khác đôi chút giữa máy | Xác minh `EXPERIMENT_MANIFEST.json`, SHA-256 dữ liệu, phiên bản gói và CPU/GPU |

## P. Phạm vi kết luận

Kết quả chỉ mô tả hiệu quả trên snapshot dữ liệu, horizon một phiên, feature và siêu tham số đã công bố. Không suy rộng thành hiệu quả đầu tư thực tế. Để mở rộng sang Bài toán 2, cần tạo volatility target tương lai; sang Bài toán 3, cần nhãn rủi ro tương lai và mô hình classification với macro-F1/balanced accuracy. Cả hai có thể làm được, nhưng cần thiết kế thí nghiệm độc lập để không làm lẫn mục tiêu với Bài toán 1.

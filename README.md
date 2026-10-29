# Dự báo tỷ suất sinh lợi và độ biến động cổ phiếu ngân hàng Việt Nam bằng Mamba đa biến

Kho mã nguồn phục vụ đề tài nghiên cứu khoa học:

> **Ứng dụng mô hình Mamba đa biến trên dữ liệu đa nguồn từ thị trường và tâm lý tin tức trong dự báo tỷ suất sinh lợi và độ biến động cổ phiếu ngành ngân hàng Việt Nam**

Dự án xây dựng một quy trình thực nghiệm có thể tái lập để đánh giá hai vấn đề:

1. LSTM và Selective State Space Model lấy cảm hứng từ Mamba có dự báo tốt hơn các baseline đơn giản hay không;
2. dữ liệu cảm xúc tin tức có tạo giá trị tăng thêm so với dữ liệu thị trường hay không.

Đây là dự án nghiên cứu, không phải hệ thống khuyến nghị giao dịch hay tư vấn đầu tư.

## 1. Phạm vi nghiên cứu

### Đối tượng

Tám cổ phiếu ngân hàng Việt Nam được sử dụng:

`VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB`.

VN-Index được dùng làm biến đại diện cho diễn biến thị trường chung.

### Bài toán dự báo

- **Bài toán 1 — tỷ suất sinh lợi:** dự báo log-return phiên kế tiếp; horizon 5 phiên được dùng như phân tích bổ sung.
- **Bài toán 2 — độ biến động:** dự báo realized volatility của 5 phiên tiếp theo.

### Khoảng dữ liệu

- Dữ liệu thị trường: từ `02/01/2020` đến `12/10/2026`.
- Dữ liệu tin tức: từ `01/01/2020` đến `12/10/2026`.
- Snapshot tin sau lọc: **14.090 tiêu đề**.

## 2. Thiết kế nghiên cứu

Nghiên cứu thực hiện đồng thời hai trục so sánh. So sánh kiến trúc dùng cùng một tập đầu vào để đối chiếu LSTM với Mamba-style. So sánh dữ liệu giữ nguyên kiến trúc và thay đổi giữa `Market stationary` và `Market stationary + News`.

Sơ đồ quy trình và giải thích các phép so sánh: [Thiết kế nghiên cứu](docs/THIET_KE_NGHIEN_CUU.md).

| Thành phần | Cấu hình |
|---|---|
| Baseline hằng | Zero return, Train Mean |
| Baseline học máy | Ridge; log-Ridge cho volatility |
| Mạng đối chứng | LSTM |
| Mạng nghiên cứu | Selective SSM (Mamba-style) |
| Dữ liệu đầu vào | Market stationary; Market stationary + News |
| Cửa sổ đầu vào | 30 phiên |
| Số lần khởi tạo | 5 seed: `42, 52, 62, 72, 82` |
| Chia tập | Train đến 31/12/2023; validation năm 2024; test từ 01/01/2025 |

Mỗi mã được huấn luyện riêng. Thiết kế hiện tại không có ticker embedding, do đó không thực hiện ablation cho thành phần này.

## 3. Dữ liệu và đặc trưng

### 3.1 Dữ liệu thị trường

Dữ liệu OHLCV và VN-Index được cache dưới dạng CSV. Cấu hình chính không đưa trực tiếp mức giá tuyệt đối vào mô hình mà sử dụng các biến có tính ổn định hơn:

- log-return;
- return trong phiên và qua đêm;
- thay đổi khối lượng và z-score khối lượng 20 phiên;
- biên độ high–low;
- volatility lịch sử 5 và 20 phiên;
- momentum 5 và 20 phiên;
- return và momentum VN-Index;
- return tương đối so với VN-Index.

### 3.2 Dữ liệu tin tức

Snapshot metadata tin được truy hồi từ Google News RSS theo tên từng ngân hàng. Dự án chỉ công bố tiêu đề, nguồn, URL và thời gian xuất bản; không tải hoặc phân phối lại toàn văn bài báo. Mã thu thập dữ liệu không thuộc phạm vi kho nghiên cứu này; snapshot, manifest và checksum cần thiết để tái lập thí nghiệm đã được lưu trong `data/`.

Dự án triển khai hai bộ chấm cảm xúc để có đối chứng:

- baseline từ điển tài chính tiếng Việt `vi_finance_lexicon_v1_title` trong `data/news.csv`;
- PhoBERT (`vinai/phobert-base-v2`) fine-tune trên tập sentiment của `undertheseanlp/UTS2017_Bank`, sinh `data/news_phobert.csv`.

Baseline từ điển thực hiện:

1. chuẩn hóa Unicode và chữ thường;
2. dò các cụm từ tài chính tích cực và tiêu cực;
3. cộng trọng số;
4. chuẩn hóa điểm về `[-1, 1]`.

Với PhoBERT, đầu ra ba lớp được đổi thành điểm liên tục
`P(positive) - P(negative)` trong `[-1,1]`. UTS2017_Bank gồm phản hồi dịch vụ
ngân hàng chứ không phải tiêu đề chứng khoán, nên kết quả vẫn phải được kiểm
định trên mẫu tiêu đề do nhóm gán nhãn. Ma trận chính trong `results/` đã được
chạy bằng `data/news_phobert.csv`; kết quả từ điển trước đó được lưu riêng tại
`results/lexicon/` để đối chứng.

Hai đặc trưng tin tức được đưa vào mô hình là:

- `sentiment_score`: điểm cảm xúc trung bình theo mã và phiên;
- `news_count`: số tiêu đề theo mã và phiên.

Tin từ 15:00 trở đi, tin cuối tuần và tin ngày nghỉ được chuyển sang phiên giao dịch kế tiếp. Quy tắc này ngăn thông tin xuất hiện sau thời điểm dự báo bị đưa ngược vào quá khứ.

Chi tiết về nguồn, schema, giấy phép và quy trình kiểm tra sentiment được trình bày tại [đặc tả dữ liệu tin tức](docs/NEWS_DATA_SCHEMA.md).

## 4. Biến mục tiêu

Với giá đóng cửa $P_t$ và log-return $r_t=\ln(P_t/P_{t-1})$, mục tiêu return là:

$$
y_{t,h}^{(r)}=\sum_{j=1}^{h}r_{t+j},\qquad h\in\{1,5\}.
$$

Mục tiêu volatility 5 phiên là:

$$
y_t^{(\sigma)}=\sqrt{\frac{1}{5}\sum_{j=1}^{5}r_{t+j}^{2}}.
$$

Mẫu được chia tập theo ngày cuối của target. Scaler, tham số mô hình và mọi lựa chọn siêu tham số chỉ sử dụng train hoặc validation; test chỉ dùng để đánh giá cuối cùng.

## 5. Mô hình và cơ chế chống overfitting

- **Train Mean:** dự báo bằng trung bình target trên train; đây là mốc của $R^2_{OOS}$.
- **Zero return:** dự báo return bằng 0.
- **Ridge:** trải phẳng cùng cửa sổ đầu vào và chọn alpha hoàn toàn trên validation. Với volatility, Ridge học trên log-target để bảo đảm dự báo dương.
- **LSTM:** mô hình tuần tự đối chứng.
- **Mamba-style:** Selective SSM tối giản với tham số chuyển trạng thái phụ thuộc đầu vào và causal convolution. Đây không phải triển khai chính thức của gói `mamba_ssm`.

Cấu hình mạng sử dụng Huber loss cho return, MSE cho volatility, AdamW với weight decay `1e-4`, early stopping và forecast shrinkage về Train Mean. Hệ số shrinkage chỉ được ước lượng trên validation và bị giới hạn trong `[0,1]`; dữ liệu test không tham gia lựa chọn hệ số.

## 6. Chỉ số đánh giá

Cả hai bài toán báo cáo RMSE, MAE, $R^2_{test}$ và $R^2_{OOS}$. Return báo cáo thêm Directional Accuracy; volatility báo cáo thêm QLIKE.

$$
R^2_{OOS}=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\hat y_i^{TrainMean})^2}.
$$

- $R^2_{OOS}>0$: mô hình tốt hơn Train Mean trên test.
- $R^2_{OOS}=0$: tương đương Train Mean.
- $R^2_{OOS}<0$: mô hình kém Train Mean.

Kết quả được tổng hợp theo từng mã, trung bình và độ lệch chuẩn qua 5 seed, sau đó trung bình trên tám mã. Kiểm định HAC so sánh squared loss của từng cấu hình với Train Mean trên cùng ngày test.
Các so sánh Mamba-style với LSTM và Market + News với Market stationary cũng được
ghép theo cùng mã, seed và ngày test. Nhiều seed của một mã dùng chung chuỗi test,
vì vậy số cặp mã–seed không được diễn giải thành số quan sát độc lập. Các p-value
theo từng cặp chưa điều chỉnh cho nhiều phép thử.

## 7. Kết quả chính

Ma trận chính gồm **568 lượt chạy** và **249.352 dự báo**. Toàn bộ 31 kiểm thử tự động của kho công khai đều đạt.

### 7.1 Return một phiên

| Mô hình | Tập đặc trưng | RMSE | MAE | $R^2_{OOS}$ | DA |
|---|---|---:|---:|---:|---:|
| Train Mean | Không áp dụng | **0,018921** | 0,012921 | 0,000000 | 0,462415 |
| LSTM | Market stationary | 0,018954 | 0,012962 | -0,003499 | 0,466629 |
| LSTM | Market stationary + PhoBERT News | 0,018959 | 0,012971 | -0,003839 | 0,466059 |
| Mamba-style | Market stationary | 0,018934 | 0,012939 | -0,001434 | 0,467084 |
| Mamba-style | Market stationary + PhoBERT News | 0,018930 | 0,012934 | -0,000890 | 0,466002 |

Mamba-style với PhoBERT tiến gần Train Mean nhưng chưa vượt baseline trên trung bình toàn nhóm. Vì vậy chưa có cơ sở kết luận mô hình dự báo được return một phiên tốt hơn dự báo hằng đơn giản.

### 7.2 Return năm phiên

| Mô hình | Tập đặc trưng | RMSE | MAE | $R^2_{OOS}$ | DA |
|---|---|---:|---:|---:|---:|
| Train Mean | Không áp dụng | **0,042155** | 0,029591 | 0,000000 | 0,520216 |
| LSTM | Market stationary | 0,042368 | 0,029718 | -0,010051 | 0,516287 |
| LSTM | Market stationary + PhoBERT News | 0,042259 | 0,029688 | -0,004518 | 0,522096 |
| Mamba-style | Market stationary | 0,042262 | 0,029652 | -0,004833 | 0,520501 |
| Mamba-style | Market stationary + PhoBERT News | 0,042209 | 0,029651 | -0,002417 | 0,518280 |

Kéo dài horizon không làm xuất hiện năng lực dự báo vượt Train Mean trên trung bình nhóm. Đây là kết quả bổ sung, không thay thế bài toán return một phiên.

### 7.3 Volatility năm phiên

| Mô hình | Tập đặc trưng | RMSE | MAE | $R^2_{OOS}$ | QLIKE |
|---|---|---:|---:|---:|---:|
| Train Mean | Không áp dụng | 0,010491 | 0,008567 | 0,000000 | 0,692076 |
| LSTM | Market stationary | **0,009671** | 0,007276 | **0,147791** | **0,689216** |
| LSTM | Market stationary + PhoBERT News | 0,009776 | **0,007209** | 0,129236 | 0,761927 |
| Mamba-style | Market stationary | 0,009772 | 0,007333 | 0,128632 | 0,730341 |
| Mamba-style | Market stationary + PhoBERT News | 0,009826 | 0,007293 | 0,119319 | 0,783877 |

Volatility là bài toán có kết quả tích cực. LSTM Market stationary đạt RMSE thấp nhất và $R^2_{OOS}=0,147791$. Mamba-style cũng vượt Train Mean, nhưng không vượt LSTM ở cấu hình chính.

So sánh ghép cặp trực tiếp trên cùng mã, seed và ngày test cho thấy Mamba-style
tốt hơn LSTM ở 21/40 cặp return h1 với Market stationary, nhưng chỉ 12/40 cặp
volatility h5. Xem [bảng so sánh kiến trúc](results/architecture_tests.csv) và
phần diễn giải trong [báo cáo đầy đủ](docs/BAO_CAO_NCKH.md).

### 7.4 Giá trị tăng thêm của tin tức

| Target | Mô hình | Số cặp news tốt hơn | Tổng số cặp |
|---|---|---:|---:|
| Return h1 | LSTM | 19 | 40 |
| Return h1 | Mamba-style | 19 | 40 |
| Return h5 | LSTM | 18 | 40 |
| Return h5 | Mamba-style | 21 | 40 |
| Volatility h5 | LSTM | 16 | 40 |
| Volatility h5 | Mamba-style | 17 | 40 |

Tin tức không tạo cải thiện ổn định trên đa số mã và seed. PhoBERT cũng không
thắng baseline từ điển một cách nhất quán: Mamba+PhoBERT tốt hơn Mamba+từ điển
15/40 cặp ở return h1, 10/40 ở return h5 và 17/40 ở volatility. Kết quả này chỉ
phản ánh snapshot và cách fine-tune hiện tại; nó không chứng minh tin tức nói
chung không có giá trị dự báo.

## 8. Cấu trúc kho mã

```text
config.py                  Cấu hình mã cổ phiếu, split, target và feature set
phobert_sentiment.py       Fine-tune PhoBERT và chấm sentiment tiêu đề
compare_sentiment_methods.py So sánh PhoBERT, từ điển và Market-only
evaluate_sentiment.py      Đánh giá sentiment trên mẫu gán nhãn thủ công
preprocess.py              Đồng bộ dữ liệu, tạo feature/target và chuỗi đầu vào
baselines.py               Zero, Train Mean và Ridge
models/lstm.py             Mô hình LSTM
models/mamba_model.py      Selective SSM Mamba-style
train.py                   Huấn luyện và đánh giá một cấu hình
run_experiments.py         Điều phối ma trận thí nghiệm
evaluate.py                Metric, bảng tổng hợp và biểu đồ
generate_report.py         Sinh báo cáo kết quả từ CSV
verify_project.py          Kiểm tra dữ liệu, schema, hash và số lượt chạy
tests/                     Kiểm thử tự động
docs/                       Báo cáo, sơ đồ và tài liệu phương pháp
results/                   Kết quả tổng hợp được công bố
```

## 9. Cài đặt

Các snapshot dữ liệu đã có sẵn nên môi trường huấn luyện không cần Vnstock hoặc API key.

```powershell
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên macOS/Linux: `python3 -m venv .venv`, sau đó chạy
`.venv/bin/python -m pip install -r requirements.txt`. Các lệnh bên dưới dùng
Python trong môi trường ảo vừa tạo. Nếu nhóm đã có môi trường riêng, có thể
thay đường dẫn interpreter tương ứng.

## 10. Chạy thực nghiệm

### Dữ liệu đầu vào

Toàn bộ OHLCV, VN-Index, metadata tin, sentiment từ điển và sentiment PhoBERT
đã nằm trong `data/`. Kho mã tập trung vào xử lý, mô hình hóa và đánh giá; code
truy xuất dữ liệu nguồn không được phân phối trong repository.

Fine-tune và chấm toàn bộ tiêu đề bằng PhoBERT:

```powershell
& .\.venv\Scripts\python.exe phobert_sentiment.py train
& .\.venv\Scripts\python.exe phobert_sentiment.py score
```

Checkpoint được lưu tại `artifacts/phobert_sentiment/` và không commit vì dung
lượng lớn. Metric sentiment được lưu tại
`results/phobert_sentiment_metrics.json`. Máy chỉ có CPU vẫn chạy được nhưng sẽ
chậm; nên dùng GPU trên Google Colab nếu có.

Kiểm tra file đã chấm và tái tạo manifest:

```powershell
& .\.venv\Scripts\python.exe phobert_sentiment.py audit
```

### Chạy ma trận chính

```powershell
& .\.venv\Scripts\python.exe run_experiments.py
```

Tiếp tục một lần chạy bị gián đoạn:

```powershell
& .\.venv\Scripts\python.exe run_experiments.py --resume
```

### Sinh báo cáo và xác minh

```powershell
& .\.venv\Scripts\python.exe generate_report.py --window 30
& .\.venv\Scripts\python.exe compare_sentiment_methods.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe verify_project.py --require-results
```

Hướng dẫn chi tiết được cung cấp tại [hướng dẫn A–Z](docs/HUONG_DAN_A_Z.md). Công thức và đặc tả phương pháp nằm trong [tài liệu phương pháp](docs/CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md).

## 11. Đầu ra công bố

- `results/metrics.csv`: metric của từng lần chạy;
- `results/predictions.csv`: toàn bộ 249.352 dự báo test theo ngày dùng để tái lập metric và kiểm định;
- `results/per_ticker_summary.csv`: kết quả theo mã và độ lệch chuẩn seed;
- `results/group_summary.csv`: kết quả toàn nhóm;
- `results/statistical_tests.csv`: kiểm định HAC so với Train Mean;
- `results/news_value_tests.csv`: so sánh ghép cặp Market + News với Market-only;
- `results/architecture_tests.csv`: so sánh ghép cặp Mamba-style với LSTM;
- `results/sentiment_method_comparison.csv`: so sánh PhoBERT với từ điển và Market-only;
- `results/phobert_sentiment_metrics.json`: đánh giá PhoBERT trên UTS2017_Bank;
- `results/lexicon/`: kết quả đối chứng dùng sentiment từ điển;
- `results/BAO_CAO_KET_QUA.md`: báo cáo tự sinh từ kết quả;
- `results/EXPERIMENT_MANIFEST.json`: cấu hình, môi trường và checksum;
- `results/bieu_do/tong_hop/`: biểu đồ cấp nhóm.

Thư mục `data/` được lưu trong repository để người đọc có thể tái lập chính xác
lần thực nghiệm đã công bố. Dữ liệu bao gồm OHLCV, VN-Index, metadata/tiêu đề
tin, điểm sentiment và các manifest checksum. Việc đưa snapshot vào kho mã
không chuyển giao quyền sở hữu đối với dữ liệu của bên thứ ba; người sử dụng
phải tuân thủ điều khoản của nguồn KBS/Vnstock, Google News và nhà xuất bản gốc.
Xem [ghi chú nguồn dữ liệu và bên thứ ba](docs/THIRD_PARTY_NOTICES.md).

Dự báo test theo từng ngày được commit để người đọc có thể kiểm tra lại metric
và kiểm định mà không cần huấn luyện lại 568 lượt. Các biểu đồ từng mã–seed và
checkpoint PhoBERT không được commit vì có thể tái tạo và có dung lượng lớn;
biểu đồ tổng hợp dùng trong báo cáo vẫn được công bố.

## 12. Hạn chế và hướng phát triển

- Kết quả dự báo chính đã dùng PhoBERT; kết quả từ điển được giữ làm đối chứng.
- UTS2017_Bank là phản hồi khách hàng, lệch lớp neutral và khác miền tiêu đề chứng khoán.
- PhoBERT đạt Accuracy 94,53% nhưng Macro-F1 63,42% và không dự đoán đúng mẫu neutral nào trên test UTS2017_Bank; Accuracy đơn lẻ vì vậy dễ gây hiểu nhầm.
- Chất lượng cả từ điển và PhoBERT cần được xác nhận trên cùng mẫu gán nhãn thủ công bằng Accuracy và Macro-F1.
- Snapshot Google News có thể thay đổi theo chỉ mục của nhà cung cấp.
- Return được tính từ cột `close` trong snapshot; nghiên cứu chưa xác nhận riêng
  việc điều chỉnh cổ tức/chia tách, nên không gọi đó là total return.
- Thời điểm xuất bản trong snapshot RSS không chứng minh tiêu đề đã có sẵn tại
  đúng thời điểm lịch sử; cần dữ liệu point-in-time để xác nhận kết quả như một
  backtest có thể giao dịch thực tế.
- Mamba-style là mô hình Selective SSM tối giản, không phải bản benchmark chính thức.
- Return ngắn hạn vẫn chưa vượt Train Mean trên trung bình nhóm.
- Các hướng mở rộng hợp lý gồm bổ sung nhãn tiêu đề chứng khoán để thích nghi miền cho PhoBERT, mô hình pooled nhiều mã, walk-forward validation và kiểm tra độ nhạy đã đăng ký trước.

## 13. Việc cần hoàn tất trước bản nộp cuối

1. Hai người gán nhãn độc lập 90 tiêu đề trong `data/news_validation_sample.csv`,
   thống nhất nhãn bất đồng và báo cáo Accuracy, Macro-F1, ma trận nhầm lẫn của
   PhoBERT và từ điển trên đúng mẫu này. Hiện cột `human_label` còn trống.
2. Xác nhận quyền công bố các snapshot OHLCV và metadata tin theo điều khoản
   của từng nguồn. Ghi lại bằng chứng hoặc giới hạn phạm vi chia sẻ nếu cần.
3. Khi trình bày, gọi kết quả tin tức là đánh giá hồi cứu theo thời điểm xuất bản;
   chỉ gọi là backtest thời gian thực khi có dữ liệu point-in-time.

## 14. Tài liệu nghiên cứu

Báo cáo đầy đủ: [docs/BAO_CAO_NCKH.md](docs/BAO_CAO_NCKH.md).

Khi sử dụng kết quả, cần công bố đầy đủ baseline, kết quả âm và giới hạn của nguồn tin; không lựa chọn riêng mã hoặc seed thuận lợi.

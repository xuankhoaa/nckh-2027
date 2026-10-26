# NCKH 2027 — Mamba đa biến cho cổ phiếu ngân hàng Việt Nam

Đề tài nghiên cứu ảnh hưởng của dữ liệu thị trường và tâm lý tin tức tới hai bài toán:

1. dự báo log-return cổ phiếu phiên kế tiếp;
2. dự báo độ biến động thực hiện trong 5 phiên kế tiếp.

Đối tượng gồm `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB`. Đây là pipeline nghiên cứu có thể tái lập, không phải công cụ tư vấn đầu tư.

| Hạng mục | Trạng thái |
|---|---|
| Ma trận thực nghiệm chính | Hoàn thành 344 lượt chạy |
| Hai bài toán đầu ra | Return 1 phiên và volatility 5 phiên |
| Dữ liệu đầu vào | Thị trường và 14.090 tiêu đề tin đã lọc |
| Kiểm thử | 19/19 kiểm thử đạt |
| Kết quả công bố | Bảng tổng hợp, kiểm định, manifest và biểu đồ đã lưu trong `results/` |

Tài liệu sử dụng và bàn giao:

- [Hướng dẫn chạy từ A–Z](HUONG_DAN_A_Z.md)
- [Đặc tả phương pháp và công thức](CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md)
- [Quy chuẩn dữ liệu tin tức](NEWS_DATA_SCHEMA.md)
- [Báo cáo nghiên cứu](bao_cao/BAO_CAO_NCKH.md)
- [Báo cáo kết quả tự sinh](results/BAO_CAO_KET_QUA.md)

## Trả lời ngắn cho câu hỏi “ablation hay chỉ so sánh mô hình?”

Cần **cả hai**, vì chúng trả lời hai câu hỏi khác nhau:

| Trục thí nghiệm | So sánh | Câu hỏi được trả lời |
|---|---|---|
| Mô hình | LSTM với Selective SSM trên đúng cùng đầu vào | Kiến trúc nào tốt hơn? |
| Dữ liệu/ablation | Market-only với Market+News trên đúng cùng mô hình | Tin tức có tạo thêm giá trị dự báo không? |
| Thành phần tin | bỏ sentiment hoặc bỏ news count | Thành phần tin nào đóng góp? |
| Baseline | mạng nơ-ron với Train Mean | Mô hình có vượt ngưỡng tối thiểu hay không? |

Không thực hiện ablation embedding mã cổ phiếu ở thiết kế chính, vì mô hình hiện được huấn luyện riêng theo từng mã và không có ticker embedding. Chỉ nên kiểm tra thành phần này nếu chuyển sang một mô hình pooled/global dùng chung tám mã.

## Phương pháp và thiết kế thực nghiệm

### Đầu vào

`Market-only` gồm OHLCV, log-return, thay đổi khối lượng, biên độ high–low, độ biến động lịch sử 5 phiên và return VN-Index.

`Market + News` bổ sung:

- `sentiment_score`: trung bình điểm cảm xúc theo mã và phiên;
- `news_count`: số bài theo mã và phiên.

Hai ablation tùy chọn là `market_news_no_sentiment` và `market_news_no_count`. LSTM và Selective SSM luôn được chạy trên cùng tập feature để tránh đánh đồng ảnh hưởng của kiến trúc với ảnh hưởng của dữ liệu.

### Hai biến mục tiêu

Với giá đóng cửa $P_t$ và log-return $r_t=\ln(P_t/P_{t-1})$:

$$
y_t^{(return)}=r_{t+1},
$$

$$
y_t^{(vol)}=\sqrt{\frac{1}{H}\sum_{j=1}^{H}r_{t+j}^2},\qquad H=5.
$$

Mẫu volatility được gán tập theo ngày cuối của horizon, nên mọi return trong nhãn train đều kết thúc trước mốc train. Dự báo volatility âm được chặn về số dương rất nhỏ trước khi đánh giá.

### Chia dữ liệu

| Tập | Ngày của nhãn | Vai trò |
|---|---|---|
| Train | đến 31/12/2023 | fit scaler, trọng số và ngưỡng tail-loss |
| Validation | 01/01/2024–31/12/2024 | early stopping |
| Test | từ 01/01/2025 | đánh giá cuối |

Feature scaler và target scaler chỉ fit trên train. Cửa sổ không bị shuffle và target luôn nằm sau ngày cuối của input.

### Mô hình và metric

- Baseline: `train_mean` cho cả hai bài toán; `naive_zero` chỉ dùng cho return.
- Đối chứng: LSTM.
- Mô hình nghiên cứu: Selective SSM tối giản lấy cảm hứng từ Mamba.
- Return: RMSE, MAE, $R^2_{OOS}$ và Directional Accuracy.
- Volatility: RMSE, MAE, $R^2_{OOS}$ và QLIKE.
- Mỗi mạng chạy mặc định 5 seed; báo cáo mean và độ lệch chuẩn theo seed, từng mã và toàn nhóm.
- Kiểm định HAC so sánh squared loss của mỗi mạng với Train Mean trên cùng ngày test.

## Trạng thái dữ liệu và kết quả

Ma trận chính đã hoàn thành **344 lượt chạy** và 151.016 dự báo trên snapshot
14.090 tiêu đề tin. Kết quả trung bình tám mã:

| Bài toán | Mô hình | RMSE | MAE | R² OOS | Metric riêng |
|---|---|---:|---:|---:|---:|
| Return | Train Mean | **0,018918** | **0,012904** | 0,000000 | DA 0,462415 |
| Return | Mamba-style, Market-only | 0,019081 | 0,013147 | -0,017150 | DA 0,456720 |
| Return | Mamba-style, Market+News | 0,019127 | 0,013184 | -0,022165 | DA 0,459226 |
| Return | LSTM, Market-only | 0,019319 | 0,013368 | -0,044754 | DA 0,463895 |
| Return | LSTM, Market+News | 0,019228 | 0,013224 | -0,032925 | DA 0,455752 |
| Volatility | Train Mean | 0,010576 | 0,008696 | 0,000000 | QLIKE 0,692187 |
| Volatility | Mamba-style, Market-only | **0,009779** | 0,007554 | **0,142239** | **QLIKE 0,683595** |
| Volatility | Mamba-style, Market+News | 0,009905 | 0,007624 | 0,118645 | QLIKE 0,761999 |
| Volatility | LSTM, Market-only | 0,009871 | **0,007530** | 0,125247 | QLIKE 0,728266 |
| Volatility | LSTM, Market+News | 0,010110 | 0,007536 | 0,079817 | QLIKE 0,834817 |

Return vẫn là kết quả âm: mọi cấu hình mạng đều chưa vượt Train Mean theo RMSE
và $R^2_{OOS}$ vẫn âm. News giúp RMSE của LSTM giảm nhẹ nhưng chỉ tốt hơn
Market-only ở 16/40 cặp mã–seed; Mamba+News kém Mamba Market-only trung bình.

Với volatility, cấu hình tốt nhất vẫn là Mamba Market-only. Market+News làm RMSE
trung bình xấu đi ở cả LSTM và Mamba; số cặp news tốt hơn lần lượt chỉ là 14/40
và 18/40. Vì vậy snapshot và sentiment baseline hiện tại **chưa cho bằng chứng
ổn định rằng tin tức tạo giá trị dự báo tăng thêm**. Đây là kết quả nghiên cứu
hợp lệ, đồng thời chỉ ra bước tiếp theo là kiểm định/gia cố bộ chấm sentiment,
không phải loại bỏ Train Mean hay chọn riêng seed thuận lợi.

Dự án đã có `collect_news.py` để tạo snapshot tin lịch sử theo từng ngân hàng
từ Google News RSS. Chỉ metadata, tiêu đề, nguồn, URL và thời gian được lưu;
không sao chép toàn văn. Tiêu đề phải nhắc trực tiếp đúng ngân hàng, dữ liệu
được loại trùng và phải đạt độ phủ tối thiểu trong cả train/validation/test.

Sentiment hiện là một **NLP baseline từ điển tài chính tiếng Việt**, chấm trên
tiêu đề theo cùng một quy tắc cho toàn kỳ. Đây là lựa chọn minh bạch và vừa sức
cho NCKH đại học, nhưng không được gọi là nhãn cảm xúc chuẩn. Trước khi chốt bài
báo, nhóm cần gán nhãn thủ công mẫu do script sinh ra và báo cáo Accuracy/F1.

### Biểu đồ kết quả chính

![RMSE bài toán return](results/bieu_do/tong_hop/group_return_rmse_h1_w30.png)

![RMSE bài toán volatility](results/bieu_do/tong_hop/group_volatility_rmse_h5_w30.png)

## Môi trường

Dự án đang dùng dòng **Vnstock 4.x** (`vnstock 4.0.8`, `vnai 2.6.0`) và API `Quote` trong `get_data.py`. Không nâng cấp chồng Vnstock 5.x vào cùng môi trường.

Môi trường đã dùng để tạo kết quả nằm tại `%USERPROFILE%\.venv`. Khi thiết lập
trên máy mới, kiểm tra môi trường trước và không cài Vnstock 5.x chồng lên dòng
4.x. Các lệnh dưới đây luôn gọi đúng Python của môi trường này:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m pip show vnstock vnstock_data vnai
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
```

API key chỉ được lưu cục bộ theo hướng dẫn Vnstock, không ghi vào mã, lệnh, báo cáo hay cuộc trò chuyện.

## Chạy nghiên cứu

Tải/cache dữ liệu thị trường nếu cần:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Thu thập snapshot tin, chấm sentiment và kiểm tra độ phủ:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" collect_news.py
```

Mở `data/news_validation_sample.csv`, điền `human_label` bằng một trong ba giá
trị `negative`, `neutral`, `positive`, sau đó kiểm tra chất lượng:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" evaluate_sentiment.py
```

Thư mục `data/` không được commit. Giữ `data/NEWS_MANIFEST.json` cùng snapshot
nội bộ khi tái lập thí nghiệm; trên GitHub chỉ công bố mã thu thập, schema và mô
tả nguồn. Kết quả có thể thay đổi nếu chỉ mục Google News thay đổi, vì vậy ngày
thu thập và checksum luôn được ghi vào manifest.

Chạy mốc Market-only cho cả hai bài toán khi chưa có dữ liệu tin:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --feature-sets market_only
```

Sau khi chuẩn bị `data/news.csv`, chạy ma trận chính Market-only và Market+News:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py
```

Chạy thêm ablation sentiment/count:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --include-ablations
```

Độ nhạy là thí nghiệm bổ sung, nên đăng ký trước một lưới nhỏ thay vì thử đến khi có kết quả đẹp:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --include-ablations --windows 20 30 60 --tail-weights 1 2 --news-cutoff-hours 14 15 16
```

Sinh báo cáo và xác minh:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" generate_report.py --window 30
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m compileall -q .
& "$env:USERPROFILE\.venv\Scripts\python.exe" verify_project.py --require-results
```

## Đầu ra

Mỗi kết quả mang đủ định danh `target`, `horizon`, `window`, `feature_set`, `news_cutoff_hour`, `tail_weight`, `ticker`, `model`, `seed`. Các tệp chính:

- `results/metrics.csv`: metric từng lần chạy;
- `results/predictions.csv`: dự báo từng ngày;
- `results/per_ticker_summary_*.csv`: mean/độ lệch chuẩn seed theo mã;
- `results/group_summary_*.csv`: tổng hợp toàn nhóm;
- `results/KIEM_DINH_HAC_VI.csv`: kiểm định so với Train Mean;
- `results/KIEM_DINH_GIA_TRI_TIN_TUC_VI.csv`: kiểm định ghép cặp News với Market-only;
- `results/BAO_CAO_KET_QUA.md`: diễn giải tự sinh;
- `results/EXPERIMENT_MANIFEST.json`: commit, checksum, môi trường và cấu hình.

### Phạm vi công bố trên GitHub

| Được lưu trong Git | Chỉ lưu cục bộ và có thể tái tạo |
|---|---|
| Mã nguồn, kiểm thử và tài liệu | CSV giá và snapshot tin trong `data/` |
| Metric theo từng lần chạy và bảng tổng hợp | `results/predictions.csv` và bảng dự báo theo ngày |
| Kiểm định HAC, manifest và biểu đồ cấp nhóm | Checkpoint và biểu đồ của từng mã/seed |

Cách phân chia này giữ kho mã gọn, tránh phát hành lại dữ liệu tổng hợp từ nguồn
bên ngoài và vẫn bảo đảm kết quả công bố có thể kiểm tra bằng checksum cùng mã
thu thập. Không có API key hoặc thông tin đăng nhập nào được đưa vào Git.

## Cấu trúc chính

```text
config.py                 cấu hình target, feature set, ticker và split
collect_news.py           tải metadata RSS, lọc thực thể và chấm sentiment baseline
evaluate_sentiment.py     đánh giá mẫu sentiment do nhóm gán nhãn thủ công
preprocess.py             đồng bộ market/news, tạo target và cửa sổ chống leakage
train.py                  baseline, LSTM, Selective SSM và tail-weighted loss
run_experiments.py        ma trận chính, ablation và sensitivity
evaluate.py               metric, bảng tổng hợp và biểu đồ
generate_report.py        bảng tiếng Việt, kiểm định HAC và báo cáo
verify_project.py         kiểm tra schema và số lượt chạy theo manifest
NEWS_DATA_SCHEMA.md       chuẩn dữ liệu tin tức
tests/                    kiểm tra target, leakage, cutoff tin, model và metric
```

## Diễn giải khoa học

Train Mean là baseline bắt buộc nhưng không phải tiêu chuẩn duy nhất. Một tuyên bố “Mamba tốt hơn” chỉ có cơ sở khi Mamba giảm sai số so với Train Mean và LSTM, kết quả ổn định qua seed/mã, và chênh lệch loss có bằng chứng thống kê hợp lý. Tương tự, chỉ được kết luận “tin tức có ích” khi cùng một kiến trúc với Market+News tốt hơn chính nó ở Market-only; việc so Mamba+News với LSTM Market-only không tách được nguyên nhân cải thiện.

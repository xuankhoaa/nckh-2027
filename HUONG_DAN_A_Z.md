# Hướng dẫn A–Z

## 1. Đề tài đang kiểm tra gì?

Hai bài toán được chạy độc lập:

- `return`: dự báo log-return phiên kế tiếp;
- `volatility`: dự báo realized volatility của 5 phiên kế tiếp.

Hai kiểu so sánh bắt buộc:

- LSTM với Mamba-style Selective SSM trên cùng feature;
- Market-only với Market+News trên cùng kiến trúc.

`train_mean` là baseline chung. `naive_zero` chỉ dùng cho return. Ablation tùy chọn lần lượt bỏ sentiment và số lượng tin. Không có ticker embedding trong mô hình riêng từng mã, nên không thực hiện ablation đó.

## 2. Môi trường

Kho dùng Vnstock 4.x trong môi trường riêng. Không cài Vnstock 5.x đè lên môi trường này.

Kết quả hiện tại được tạo bằng môi trường `%USERPROFILE%\.venv`. Trên máy mới,
kiểm tra môi trường trước khi cài để không ghi đè một thế hệ Vnstock khác:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m pip show vnstock vnstock_data vnai
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
```

Không gửi API key qua chat và không commit `.env`.

## 3. Dữ liệu thị trường

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
```

Mỗi CSV cần `time,open,high,low,close,volume`. Pipeline kiểm tra thiếu cột, trùng ngày, giá trị không hữu hạn, quy tắc OHLC và checksum.

## 4. Dữ liệu tin tức

Tạo snapshot từ Google News RSS bằng tên đầy đủ của tám ngân hàng:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" collect_news.py
```

Script quét theo tháng, tự chia nhỏ khoảng chạm giới hạn kết quả, bắt buộc tên
ngân hàng xuất hiện trong tiêu đề, loại trùng và tạo:

- `data/news_raw.csv`: metadata tin đã chuẩn hóa;
- `data/news.csv`: thêm điểm sentiment và nhãn;
- `data/NEWS_MANIFEST.json`: truy vấn, độ phủ, ngày lấy và checksum;
- `data/news_validation_sample.csv`: mẫu để nhóm kiểm tra thủ công.

Ba cột tối thiểu mà mô hình sử dụng là:

```text
published_at,ticker,sentiment_score
```

Tin sau cutoff 15:00 và tin cuối tuần được chuyển sang phiên kế tiếp. Dòng không
có tin nhận `sentiment_score=0` và `news_count=0`. Mỗi mã phải có ít nhất 20 bài
trong từng tập train/validation/test; nếu không, pipeline dừng trước khi xóa kết
quả cũ.

Điền cột `human_label` của mẫu bằng `negative`, `neutral` hoặc `positive`, rồi chạy:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" evaluate_sentiment.py
```

Ghi Accuracy, macro-F1 và ma trận nhầm lẫn vào phụ lục. Nếu chất lượng thấp, sửa
từ điển dựa trên lỗi trong train/validation; không dùng kết quả test dự báo để
chọn lại từ điển. Chi tiết xem [NEWS_DATA_SCHEMA.md](NEWS_DATA_SCHEMA.md).

## 5. Target và leakage

Với input kết thúc tại phiên t:

- return target là $r_{t+1}$;
- volatility target là $\sqrt{H^{-1}\sum_{j=1}^{H}r_{t+j}^2}$, mặc định `H=5`.

Train kết thúc 31/12/2023, validation là năm 2024, test từ 01/01/2025. Mẫu được chia theo ngày cuối target, không phải ngày đầu input. Scaler và ngưỡng 90% cho tail-weighted loss chỉ fit trên train.

## 6. Chạy thử một cấu hình

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" train.py --model mamba --ticker VCB --target return --feature-set market_only --seed 42 --epochs 5
```

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" train.py --model lstm --ticker VCB --target volatility --feature-set market_news --seed 42 --epochs 5
```

## 7. Chạy các ma trận

Market-only, hai bài toán:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --feature-sets market_only
```

Ma trận chính sau khi có news:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py
```

Thêm ablation:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --include-ablations
```

Sensitivity có kiểm soát:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" run_experiments.py --include-ablations --windows 20 30 60 --tail-weights 1 2 --news-cutoff-hours 14 15 16
```

Lưới sensitivity làm số lượt chạy tăng mạnh. Nên chạy ma trận chính trước; chỉ chạy lưới đã đăng ký trước, không chọn lại tham số bằng test.

## 8. Sinh và đọc báo cáo

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" generate_report.py --window 30
```

- Return: đọc RMSE, MAE, R² OOS và DA.
- Volatility: đọc RMSE, MAE, R² OOS và QLIKE.
- RMSE/MAE/QLIKE càng thấp càng tốt; R² càng cao càng tốt.
- Luôn xem mean ± std qua seed, từng mã và toàn nhóm.
- Kiểm định HAC dùng Train Mean làm mốc; cải thiện MSE dương nghĩa là mạng tốt hơn.

## 9. Kiểm tra trước khi nộp

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m compileall -q .
& "$env:USERPROFILE\.venv\Scripts\python.exe" verify_project.py --require-results
```

Nếu `results/` vẫn là pilot schema cũ, kiểm tra có `--require-results` sẽ yêu cầu chạy lại. Mặc định `run_experiments.py` chỉ xóa output tái tạo được, không xóa dữ liệu gốc.

## 10. Câu kết luận được phép viết

- “Mamba vượt Train Mean” chỉ khi RMSE thấp hơn và ổn định qua mã/seed.
- “News có ích” chỉ khi cùng mô hình, cùng seed, Market+News tốt hơn Market-only.
- Nếu Mamba/LSTM kém Train Mean, báo cáo đó là kết quả âm; không xóa baseline hoặc chọn riêng seed đẹp.
- Pilot OHLCV cũ không phải kết quả cuối của đề tài hai đầu ra đa nguồn.

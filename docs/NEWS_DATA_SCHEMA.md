# Dữ liệu tin tức: nguồn, schema và kiểm soát chất lượng

## Nguồn mặc định

Snapshot metadata tin lịch sử được truy hồi từ **Google News RSS**, theo tên
thương hiệu của từng ngân hàng. Đây là công cụ tổng hợp; trường `source` và
`source_url` cho biết nhà xuất bản gốc. Dữ liệu chỉ lưu tiêu đề, nguồn, URL và
thời điểm, không chứa toàn văn bài báo. Code truy xuất nguồn không thuộc phạm vi
kho nghiên cứu; snapshot và manifest kiểm chứng đã được công bố trong `data/`.

Lý do chọn phương án này:

- miễn phí, không cần API key và chạy được trong môi trường hiện tại;
- có thể truy vấn theo khoảng ngày từ 2020;
- giữ được URL, nguồn và thời gian để kiểm tra;
- vừa sức cho đề tài đại học.

Nhược điểm phải ghi trong báo cáo: chỉ mục của Google có thể thay đổi; độ phủ
giữa ngân hàng và giai đoạn không đồng đều; truy vấn có thể có nhiễu. Vì vậy mã
bắt buộc tên/mã ngân hàng xuất hiện ngay trong tiêu đề, loại trùng và kiểm tra độ
phủ riêng train/validation/test.

## Snapshot được công bố

Các tệp được tạo trong `data/`:

| Tệp | Nội dung |
|---|---|
| `news_raw.csv` | metadata và tiêu đề sau lọc thực thể/loại trùng |
| `news.csv` | dữ liệu trên cộng sentiment tự động |
| `news_phobert.csv` | dữ liệu trên cộng xác suất và sentiment PhoBERT |
| `NEWS_PHOBERT_MANIFEST.json` | checksum, phân bố nhãn, thống kê điểm và metric PhoBERT |
| `NEWS_MANIFEST.json` | truy vấn, số yêu cầu, độ phủ, thời điểm lấy, checksum |
| `news_validation_sample.csv` | mẫu phân tầng để gán nhãn thủ công |

Snapshot và manifest hiện được lưu trong repository để tái lập đúng lần chạy đã
công bố. Các bản ghi chỉ chứa metadata, tiêu đề và liên kết, không chứa toàn văn.
Quyền đối với tiêu đề, bài viết và metadata gốc vẫn thuộc các nhà cung cấp/nhà
xuất bản tương ứng; việc sử dụng lại phải tuân thủ điều khoản của nguồn.

## Schema dùng cho mô hình

Ba cột bắt buộc:

| Cột | Kiểu | Quy tắc |
|---|---|---|
| `published_at` | ISO-8601 datetime | Có múi giờ; dữ liệu mặc định quy về giờ Việt Nam |
| `ticker` | chuỗi | Một trong `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB` |
| `sentiment_score` | số thực | Trong `[-1,1]`; âm/0/dương tương ứng tiêu cực/trung tính/tích cực |

Các cột truy xuất nguồn nên giữ: `title`, `source`, `source_url`, `url`, `query`,
`retrieved_at`, `sentiment_method`. `news_count` không cần có sẵn; pipeline tự
đếm bài theo mã và phiên.

## Cách chấm sentiment

### Baseline từ điển

Phiên bản mặc định `vi_finance_lexicon_v1_title` là một NLP baseline từ điển:

1. chuẩn hóa Unicode và chữ thường;
2. khớp các cụm tài chính tương đối rõ nghĩa như “lợi nhuận tăng”, “giảm nợ
   xấu”, “lợi nhuận giảm”, “bị xử phạt”, “khởi tố”;
3. cộng trọng số tích cực/tiêu cực;
4. chuẩn hóa chênh lệch về `[-1,1]`; không có tín hiệu rõ ràng thì bằng 0.

Ưu điểm là minh bạch, không cần mô hình lớn và áp dụng nhất quán toàn kỳ. Hạn
chế là không hiểu đầy đủ ngữ cảnh. Vì vậy đây không phải “nhãn thật” và phải có
kiểm tra thủ công.

### PhoBERT

`phobert_sentiment.py` fine-tune `vinai/phobert-base-v2` trên cấu hình
`sentiment` của `undertheseanlp/UTS2017_Bank`. Train gốc được tách tiếp thành
train/validation có phân tầng; test chính thức chỉ dùng đánh giá cuối. Hàm mất
mát dùng trọng số lớp theo căn bậc hai nghịch đảo tần suất để hạn chế lệch lớp
neutral mà không khuếch đại quá mức số ít mẫu neutral.

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" phobert_sentiment.py train
& "$env:USERPROFILE\.venv\Scripts\python.exe" phobert_sentiment.py score
```

Mỗi tiêu đề có thêm ba xác suất, nhãn có xác suất lớn nhất, confidence và:

```text
sentiment_score = P(positive) - P(negative)
```

UTS2017_Bank có 2.471 phản hồi khách hàng (1.977 train, 494 test), không có ngày
và không phải tin thị trường. Bộ này chỉ dùng để học bộ phân loại cảm xúc; ngày,
mã và tiêu đề dự báo vẫn lấy từ `news_raw.csv`. Đây là domain adaptation gần,
không thay thế kiểm định trên tiêu đề chứng khoán.

Checkpoint được chọn ở epoch 1 theo Macro-F1 validation. Trên test chính thức,
Accuracy = 0,9453, Macro-F1 = 0,6342 và weighted-F1 = 0,9377. Mô hình dự đoán
đúng 0/8 mẫu neutral; khi chấm snapshot tin, nhãn cứng gồm 11.784 positive,
2.306 negative và 0 neutral. Do đó pipeline dự báo dùng điểm xác suất liên tục,
không coi nhãn cứng là nhãn thật. Chi tiết nằm trong
`results/phobert_sentiment_metrics.json`.

Mở `data/news_validation_sample.csv`, đọc tiêu đề và điền `human_label` bằng
`negative`, `neutral` hoặc `positive`. Sau đó chạy:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" evaluate_sentiment.py
```

Báo cáo số mẫu, Accuracy, macro-F1 và confusion matrix. Nếu hai người gán nhãn,
nên báo cáo thêm mức đồng thuận; khi bất đồng thì thảo luận để có nhãn cuối.

Để kiểm định PhoBERT, chấm chính tệp mẫu rồi giữ nguyên cột `human_label`:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" phobert_sentiment.py score `
  --input data/news_validation_sample.csv `
  --output data/news_validation_sample_phobert.csv
& "$env:USERPROFILE\.venv\Scripts\python.exe" evaluate_sentiment.py `
  --sample data/news_validation_sample_phobert.csv
```

## Đồng bộ chống leakage

- Tin trước cutoff mặc định 15:00 được gán vào phiên cùng ngày nếu có giao dịch.
- Tin từ 15:00 trở đi được gán vào phiên kế tiếp.
- Tin cuối tuần/ngày nghỉ được gán vào phiên giao dịch kế tiếp.
- Có thể kiểm tra độ nhạy với `--news-cutoff-hours 14 15 16`.
- Ngày không có tin nhận sentiment 0 và news count 0.
- Mỗi mã phải có tối thiểu 20 bài ở từng tập train, validation và test trước khi
  một thí nghiệm Market+News được phép chạy.

Không thay tin thiếu bằng dữ liệu giả, không dùng thời điểm thu thập thay cho
thời điểm xuất bản và không chọn lại từ điển/siêu tham số dựa trên kết quả test.

## Dùng nguồn khác

Nếu nhóm có quyền truy cập `vnstock_news` hoặc một bộ dữ liệu được cấp phép, có
thể thay nguồn thu thập nhưng phải đưa về đúng schema, lưu giấy phép/nguồn và
manifest, rồi chạy lại toàn bộ thí nghiệm. Không trộn hai nguồn mà không có cột
nguồn và quy tắc loại trùng rõ ràng.

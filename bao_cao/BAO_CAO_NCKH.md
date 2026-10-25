# Bản thảo báo cáo NCKH

## Tên đề tài

**Ứng dụng mô hình Mamba đa biến trên dữ liệu đa nguồn từ thị trường và tâm lý tin tức trong dự báo tỷ suất sinh lợi và độ biến động cổ phiếu ngành ngân hàng Việt Nam**

## 1. Bối cảnh

Tỷ suất sinh lợi ngắn hạn có nhiễu lớn và phụ thuộc vào thông tin mới, trong khi độ biến động có tính dai dẳng nhưng thay đổi theo trạng thái thị trường. Dữ liệu OHLCV mô tả diễn biến giao dịch đã xảy ra; tin tức có thể bổ sung tín hiệu về kỳ vọng và tâm lý chưa phản ánh đầy đủ trong lịch sử giá. Nghiên cứu vì vậy không chỉ hỏi kiến trúc Mamba có tốt hơn LSTM hay không, mà còn hỏi kênh tin tức có tạo giá trị dự báo tăng thêm hay không.

## 2. Mục tiêu và câu hỏi nghiên cứu

Nghiên cứu giải quyết hai bài toán:

1. dự báo log-return cổ phiếu ở phiên kế tiếp;
2. dự báo độ biến động thực hiện trong 5 phiên kế tiếp.

Câu hỏi nghiên cứu:

- Trên cùng dữ liệu đầu vào, Selective SSM lấy cảm hứng từ Mamba có giảm sai số so với LSTM và Train Mean không?
- Trên cùng kiến trúc, Market+News có tốt hơn Market-only không?
- Điểm sentiment và số lượng tin đóng góp như thế nào?
- Kết quả có ổn định theo mã cổ phiếu và seed hay không?

## 3. Dữ liệu

Mẫu gồm tám cổ phiếu ngân hàng `VCB`, `BID`, `CTG`, `MBB`, `TCB`, `ACB`, `VPB`, `STB`, cùng VN-Index. Snapshot thị trường hiện có từ 02/01/2020 đến 12/10/2026.

Nhóm feature thị trường gồm OHLCV, log-return, thay đổi khối lượng, biên độ high–low, realized volatility quá khứ 5 phiên và return VN-Index. Nhóm tin tức gồm sentiment trung bình và số bài theo mã/phiên. Metadata tin được truy hồi từ Google News RSS bằng tên thương hiệu của từng ngân hàng; chỉ lưu tiêu đề, nguồn, URL và thời điểm xuất bản, không lưu toàn văn. Bài chỉ được giữ khi tiêu đề nhắc trực tiếp đúng ngân hàng và được loại trùng theo URL hoặc tiêu đề–ngày.

Điểm sentiment là NLP baseline từ điển tài chính tiếng Việt trên tiêu đề, nhận giá trị trong $[-1,1]$. Phương pháp này có ưu điểm minh bạch và dễ tái lập nhưng không hiểu đầy đủ ngữ cảnh, phủ định hoặc mỉa mai. Vì vậy nghiên cứu phải báo cáo thêm kiểm tra thủ công trên mẫu phân tầng ba nhãn; không được mô tả điểm tự động là “nhãn thật”. Manifest tin lưu truy vấn, ngày thu thập, độ phủ và checksum snapshot.

Quy tắc thông tin: tin trước 15:00 có thể dùng cho dự báo được lập sau phiên đó; tin từ 15:00 trở đi và tin ngày nghỉ được chuyển sang phiên giao dịch kế tiếp. Phân tích độ nhạy có thể dùng cutoff 14:00, 15:00 và 16:00.

## 4. Biến mục tiêu

Với $r_t=\ln(P_t/P_{t-1})$:

$$
y_t^{(r)}=r_{t+1}.
$$

Với horizon $H=5$:

$$
y_t^{(\sigma)}=\sqrt{\frac{1}{H}\sum_{j=1}^{H}r_{t+j}^2}.
$$

Đây là realized volatility chưa annualize. Ngày nhãn của bài toán volatility là ngày cuối trong 5 phiên tương lai; cách gán này ngăn một nhãn train đi qua ranh giới validation/test.

## 5. Thiết kế thí nghiệm

### 5.1 Chia dữ liệu

- Train: nhãn kết thúc không muộn hơn 31/12/2023.
- Validation: nhãn trong năm 2024.
- Test: nhãn từ 01/01/2025.

Scaler feature, scaler target và ngưỡng tail loss chỉ fit trên train. Early stopping chỉ dùng validation. Test không tham gia chọn epoch hoặc siêu tham số.

### 5.2 Hai trục so sánh

| Trục | Cấu hình |
|---|---|
| Baseline | zero return (chỉ return), Train Mean |
| Kiến trúc | LSTM, Selective SSM (Mamba-style) |
| Nguồn dữ liệu | Market-only, Market+News |
| Ablation tin | bỏ sentiment, bỏ news count |

Mỗi kiến trúc được chạy trên cùng tập feature. Do mô hình chính huấn luyện riêng từng mã, nghiên cứu không dùng ticker embedding và không thực hiện ablation embedding. Nếu sau này xây dựng mô hình pooled cho tám mã, ticker embedding phải được đăng ký như một thí nghiệm riêng.

### 5.3 Lặp và độ nhạy

Mạng được chạy với các seed `42, 52, 62, 72, 82`; baseline xác định chỉ cần một lần. Cấu hình chính dùng cửa sổ 30 phiên, cutoff 15:00 và MSE thường (`tail_weight=1`). Phân tích độ nhạy được phép kiểm tra cửa sổ 20/30/60, tail weight 1/2 và cutoff 14/15/16, nhưng phải công bố cả lưới thay vì chỉ chọn cấu hình tốt trên test.

## 6. Chỉ số và kiểm định

Hai bài toán báo cáo RMSE, MAE và $R^2_{OOS}$. Return báo cáo thêm Directional Accuracy; volatility báo cáo thêm QLIKE. Mỗi bảng cần có kết quả từng mã, mean ± std giữa seed và tổng hợp toàn nhóm sau khi mỗi mã được cho trọng số bằng nhau.

Kiểm định HAC dùng chênh lệch squared loss giữa mạng và Train Mean trên cùng quan sát test. Với volatility 5 phiên chồng lấn, độ trễ HAC tối thiểu là $H-1$. p-value chỉ là bằng chứng bổ sung; kết luận phải xét cả độ lớn cải thiện và độ ổn định.

## 7. Kết quả ma trận chính

Ma trận chính đã hoàn thành 344 lượt chạy và 151.016 dự báo. Snapshot tin sau lọc gồm 14.090 tiêu đề và đạt tối thiểu hàng chục đến hàng trăm bài cho mỗi mã trong từng tập. Không seed/mã nào được loại khỏi tổng hợp.

Với return, Train Mean vẫn tốt nhất theo RMSE 0,018918. LSTM Market+News đạt 0,019228, tốt hơn nhẹ LSTM Market-only 0,019319 nhưng vẫn kém baseline và chỉ cải thiện ở 16/40 cặp mã–seed. Mamba Market+News đạt 0,019127, kém nhẹ Mamba Market-only 0,019081 và cải thiện ở 18/40 cặp. Do đó chưa có bằng chứng ổn định rằng news cải thiện dự báo return.

Với volatility, Mamba Market-only đạt RMSE nhóm 0,009779, thấp hơn LSTM Market-only 0,009871 và Train Mean 0,010576. Mamba Market-only cũng tốt nhất theo $R^2_{OOS}$ nhóm (0,142239) và QLIKE (0,683595). Khi thêm news, RMSE tăng lên 0,009905 với Mamba và 0,010110 với LSTM; số cặp news tốt hơn chỉ là 18/40 và 14/40. Như vậy các feature tin hiện tại chưa tạo giá trị tăng thêm ổn định cho volatility, dù các mạng nói chung vẫn dự báo volatility tốt hơn Train Mean theo RMSE.

## 8. Khung bảng kết quả cuối

### Bài toán return

| Mô hình | Tập đặc trưng | RMSE nhóm | MAE nhóm | $R^2_{OOS}$ | DA |
|---|---|---:|---:|---:|---:|
| Train Mean | Không áp dụng | 0,018918 | 0,012904 | 0,000000 | 0,462415 |
| LSTM | Market-only | 0,019319 | 0,013368 | -0,044754 | 0,463895 |
| LSTM | Market+News | 0,019228 | 0,013224 | -0,032925 | 0,455752 |
| Mamba | Market-only | 0,019081 | 0,013147 | -0,017150 | 0,456720 |
| Mamba | Market+News | 0,019127 | 0,013184 | -0,022165 | 0,459226 |

### Bài toán volatility

| Mô hình | Tập đặc trưng | RMSE nhóm | MAE nhóm | $R^2_{OOS}$ | QLIKE |
|---|---|---:|---:|---:|---:|
| Train Mean | Không áp dụng | 0,010576 | 0,008696 | 0,000000 | 0,692187 |
| LSTM | Market-only | 0,009871 | 0,007530 | 0,125247 | 0,728266 |
| LSTM | Market+News | 0,010110 | 0,007536 | 0,079817 | 0,834817 |
| Mamba | Market-only | 0,009779 | 0,007554 | 0,142239 | 0,683595 |
| Mamba | Market+News | 0,009905 | 0,007624 | 0,118645 | 0,761999 |

## 9. Quy tắc kết luận

- Chỉ kết luận Mamba tốt hơn khi so trên cùng feature và Mamba vượt cả LSTM lẫn Train Mean một cách ổn định.
- Chỉ kết luận news có ích khi cùng mô hình Market+News tốt hơn Market-only.
- Nếu kết quả khác nhau giữa các mã, phải nêu rõ tính không đồng nhất thay vì chỉ báo cáo trung bình nhóm.
- Nếu $R^2_{OOS}<0$ hoặc RMSE cao hơn Train Mean, ghi nhận kết quả âm; không loại baseline hoặc chọn riêng seed thuận lợi.

## 10. Trạng thái hoàn thiện

Mã nguồn cho hai target, thu thập và đồng bộ tin theo cutoff, Market-only/Market+News, hai ablation, nhiều seed, sensitivity, metric và bảng tổng hợp đã được chuẩn bị. Ma trận chính đã hoàn thành và được xác minh. Snapshot tin được tạo cục bộ theo [NEWS_DATA_SCHEMA.md](../NEWS_DATA_SCHEMA.md). Kết quả news hiện được xem là kết quả của sentiment baseline; trước khi khóa bài báo, nhóm phải hoàn tất gán nhãn mẫu kiểm tra sentiment và lưu Accuracy/F1 vào phụ lục.

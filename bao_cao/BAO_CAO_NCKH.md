# Báo cáo NCKH dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Selective State Space Model

## Tóm tắt

Nghiên cứu đánh giá khả năng dự báo log-return một phiên kế tiếp cho tám cổ phiếu ngân hàng Việt Nam từ dữ liệu OHLCV hằng ngày và return VN-Index. Dữ liệu snapshot đến 2026-10-12 được chia theo thời gian thành train, validation và test để bảo vệ tính nhân quả của dự báo. Thiết kế so sánh hai baseline hằng, LSTM, Selective SSM cơ sở và Selective SSM bổ sung return VN-Index; mỗi mạng được lặp trên năm seed. Kết quả trung bình nhóm cho thấy baseline trung bình train có RMSE thấp nhất (0.018918). Mạng tốt nhất là Selective SSM cơ sở với RMSE 0.019145, cao hơn baseline 0.000227. Vì vậy, nghiên cứu không khẳng định Selective SSM vượt baseline ở cấu hình hiện tại; kết quả âm này được báo cáo như một phát hiện cần thiết thay vì bị loại bỏ. Toàn bộ dự báo, metric, SHA-256 dữ liệu và cấu hình môi trường được lưu để tái lập.

**Từ khóa:** dự báo tỷ suất sinh lợi; cổ phiếu ngân hàng; Selective State Space Model; Mamba; LSTM; thị trường chứng khoán Việt Nam.

## 1. Vấn đề nghiên cứu và phạm vi

### 1.1. Bài toán cần giải quyết

Mục tiêu là dự báo một đại lượng liên tục: `log(close_(t+1) / close_t)`. Đây là Bài toán 1 — hồi quy tỷ suất sinh lợi, không phải dự báo giá đóng cửa. Dùng log-return thay vì giá có hai lý do: giá các mã có thang đo khác nhau, còn return biểu diễn biến động tương đối; đồng thời các log-return có thể cộng dồn theo thời gian. Dự báo horizon một phiên phù hợp với câu hỏi ngắn hạn, nhưng cũng là trường hợp rất nhiễu vì thông tin mới phát sinh sau thời điểm t chưa có trong feature.

Nghiên cứu trả lời hai câu hỏi: (i) Selective SSM có giảm sai số dự báo so với hai baseline và LSTM trên cùng dữ liệu không; (ii) thêm return VN-Index có giúp Selective SSM không. Câu trả lời chỉ áp dụng cho snapshot, feature, horizon và siêu tham số được công bố.

### 1.2. Những gì không thuộc phạm vi

Nghiên cứu không đưa ra khuyến nghị mua/bán, không backtest chiến lược hay tính chi phí giao dịch. Nó chưa tạo target độ biến động tương lai cho Bài toán 2 và chưa tạo nhãn rủi ro thấp/trung bình/cao để làm Bài toán 3. Các bài toán đó cần target, metric và thiết kế đánh giá riêng; không được gộp lẫn với hồi quy return.

## 2. Dữ liệu và lý do chọn feature

### 2.1. Dữ liệu

Tám mã VCB, BID, CTG, MBB, TCB, ACB, VPB và STB được chọn để giữ cùng bối cảnh ngành ngân hàng nhưng vẫn có khác biệt về quy mô và thanh khoản. Đầu vào gốc gồm open, high, low, close và volume; VN-Index đại diện cho trạng thái thị trường chung. Các tệp dữ liệu được kiểm tra schema, trùng ngày, giá trị thiếu/không hữu hạn, điều kiện OHLC và khối lượng trước khi dùng.

Không tự sửa dữ liệu nguồn. Manifest ghi SHA-256, số dòng và phạm vi từng CSV để người đọc có thể xác định đúng snapshot. Cảnh báo cần nêu rõ: VNINDEX: 3 dòng close nằm ngoài [low, high] từ nguồn dữ liệu; pipeline chỉ dùng close để tính vnindex_return và không tự sửa dữ liệu.

### 2.2. Feature dẫn xuất

Từ mỗi phiên t, pipeline dùng log-return cổ phiếu, thay đổi tương đối khối lượng và biên độ high-low đã chuẩn hóa. Những feature này lần lượt mô tả biến động đóng cửa, thay đổi thanh khoản và mức dao động trong phiên. Return VN-Index được thêm riêng ở biến thể cải tiến để kiểm tra giả thuyết rằng thị trường chung mang thêm thông tin cho cổ phiếu ngân hàng.

Cửa sổ 30 phiên tương đương xấp xỉ một tháng rưỡi giao dịch. Đây là lựa chọn cân bằng giữa việc giữ bối cảnh gần và việc không làm số mẫu hữu ích giảm quá nhiều; nó được khóa trước khi chạy bộ thực nghiệm chính. Kết quả không chứng minh 30 là cửa sổ tối ưu tuyệt đối; window 60/90 cần là thí nghiệm tiếp theo được lập kế hoạch trước.

## 3. Thiết kế dữ liệu để tránh rò rỉ tương lai

Mỗi mẫu có 30 feature vector từ t-29 đến t và chỉ dự báo return ở t+1. Mẫu được gán tập theo **ngày của nhãn**, không theo ngày bắt đầu cửa sổ. Train kết thúc 31/12/2023, validation là năm 2024 và test bắt đầu 01/01/2025. Chia ngẫu nhiên sẽ trộn quan sát tương lai vào giai đoạn học, tạo kết quả quá lạc quan cho chuỗi thời gian.

StandardScaler chỉ fit trên feature train. Nhãn cũng được chuẩn hóa bằng trung bình và độ lệch chuẩn của train trước khi tối ưu rồi đảo chuẩn hóa trước đánh giá. Lý do là log-return có biên độ nhỏ; chuẩn hóa giúp gradient ổn định mà không dùng thông tin validation/test. Validation chỉ chọn epoch dừng sớm; test không tham gia chọn mô hình hoặc siêu tham số.

## 4. Mô hình và lý do lựa chọn

### 4.1. Hai baseline là điều kiện bắt buộc

`naive_zero` dự báo return bằng 0 và `train_mean` dự báo trung bình return của train. Return ngày thường có trung bình gần 0, nên baseline đơn giản có thể rất mạnh. Nếu mạng không vượt baseline, không thể kết luận kiến trúc phức tạp hữu ích. Đặt baseline vào cùng tập test giúp kết quả có ý nghĩa hơn việc chỉ so sánh các mạng với nhau.

### 4.2. LSTM và Selective SSM

LSTM là đối chứng tuần tự phổ biến, được dùng để xem lợi ích có đến từ Selective SSM hay chỉ từ việc dùng một mạng chuỗi. LSTM dùng một lớp hidden size 8 và có 585 tham số.

Selective SSM dùng RMSNorm, chiếu đầu vào, causal depthwise convolution, tham số chọn lọc Delta/B/C, cổng SiLU, residual connection và trạng thái có d_state bằng 8. Cập nhật trạng thái có dạng `h_t = a_t * h_(t-1) + b_t`; mã dùng prefix scan theo phép ghép affine kết hợp để xử lý toàn bộ cửa sổ nhưng test đơn vị chứng minh kết quả khớp truy hồi tuần tự. Bản cơ sở có 545 tham số; bản thêm VN-Index có 553 tham số. Quy mô gần LSTM nhằm hạn chế kết luận bị chi phối bởi số tham số trên dữ liệu không quá lớn.

### 4.3. Vì sao tự triển khai thay vì dùng mã MambaStock

Mã dự án không fork, import hoặc sửa tiếp code MambaStock. MambaStock chỉ là tài liệu tham khảo học thuật. Selective SSM ở đây là hiện thực độc lập, tối giản theo mô tả trong bài Mamba [1], điều chỉnh cho bài toán hồi quy 30 phiên và dữ liệu Việt Nam. Cách làm này giúp kiểm soát rõ phần nào được dùng, pipeline nào tạo target và cách đánh giá nào áp dụng; đồng thời tránh trình bày nhầm mã thí nghiệm này như một bản benchmark chính thức của MambaStock hoặc thư viện `mamba_ssm`.

Do đó, tên chính xác trong báo cáo là **Selective SSM lấy cảm hứng từ Mamba**. Không nên tuyên bố đây là tái lập nguyên xi MambaStock hay kết quả so sánh trực tiếp với implementation CUDA chính thức.

### 4.4. Thiết lập huấn luyện

Adam với learning rate 0,001, tối đa 40 epoch và early stopping patience 8 được dùng nhất quán. Early stopping dựa trên validation để giảm nguy cơ chọn mô hình theo test. Năm seed 42, 52, 62, 72 và 82 được dùng vì khởi tạo mạng và tối ưu ngẫu nhiên có thể làm thay đổi kết quả; báo cáo vì vậy có cả trung bình và độ lệch chuẩn qua seed. Các hyperparameter này là cấu hình nghiên cứu, không phải kết quả của tối ưu hóa theo test.

## 5. Đánh giá và kiểm định

RMSE và MAE đo độ lớn sai số, nên càng nhỏ càng tốt. R2 cho biết phần biến thiên được mô hình giải thích; R2 âm nghĩa là kém dự báo hằng bằng trung bình của chính test theo metric này. Directional Accuracy (DA) đo đúng dấu tăng/giảm, nhưng không phản ánh độ lớn sai số; vì vậy DA cao không tự động có nghĩa mô hình tốt hơn.

Kết quả nhóm được tổng hợp theo hai tầng: trung bình seed cho từng mã trước, rồi trung bình các mã để mỗi mã có cùng trọng số. `KIEM_DINH_HAC_VI.csv` so sánh squared loss của từng seed mạng với baseline zero trên đúng các ngày test; phương sai dài hạn dùng trọng số HAC và p-value xấp xỉ chuẩn. Kiểm định này là bằng chứng bổ sung, không thay thế RMSE/MAE, kiểm tra nhiều seed và phân tích từng mã.

## 6. Kết quả thực nghiệm

| Mô hình                   | Cửa sổ phiên | Tập đặc trưng                                | Số mã | Số seed | Số tham số | RMSE trung bình | MAE trung bình | R² trung bình | DA trung bình | ĐLC RMSE giữa mã | ĐLC MAE giữa mã | ĐLC R² giữa mã | ĐLC DA giữa mã | ĐLC RMSE giữa seed | ĐLC MAE giữa seed | ĐLC R² giữa seed | ĐLC DA giữa seed |
| ------------------------- | ------------ | -------------------------------------------- | ----- | ------- | ---------- | --------------- | -------------- | ------------- | ------------- | ---------------- | --------------- | -------------- | -------------- | ------------------ | ----------------- | ---------------- | ---------------- |
| Baseline zero return      | 30           | Không áp dụng                                | 8     | 1       | 0          | 0.018928        | 0.012859       | -0.001364     | 0.078588      | 0.002384         | 0.002028        | 0.001438       | 0.028659       | 0.000000           | 0.000000          | 0.000000         | 0.000000         |
| Baseline trung bình train | 30           | Không áp dụng                                | 8     | 1       | 0          | 0.018918        | 0.012899       | -0.000482     | 0.462415      | 0.002370         | 0.002005        | 0.000441       | 0.033765       | 0.000000           | 0.000000          | 0.000000         | 0.000000         |
| LSTM                      | 30           | OHLCV và đặc trưng dẫn xuất                  | 8     | 5       | 585        | 0.019195        | 0.013171       | -0.030056     | 0.459055      | 0.002443         | 0.002043        | 0.013938       | 0.018197       | 0.000323           | 0.000339          | 0.036557         | 0.021752         |
| Selective SSM cơ sở       | 30           | OHLCV và đặc trưng dẫn xuất                  | 8     | 5       | 545        | 0.019145        | 0.013180       | -0.024805     | 0.459510      | 0.002390         | 0.002023        | 0.008905       | 0.019179       | 0.000155           | 0.000189          | 0.016868         | 0.022782         |
| Selective SSM có VN-Index | 30           | OHLCV, đặc trưng dẫn xuất và return VN-Index | 8     | 5       | 553        | 0.019156        | 0.013211       | -0.026057     | 0.461048      | 0.002433         | 0.002066        | 0.020065       | 0.016982       | 0.000354           | 0.000375          | 0.038090         | 0.022012         |

Baseline trung bình train là mô hình có RMSE thấp nhất: 0.018918. Trong ba mạng, Selective SSM cơ sở có RMSE tốt nhất: 0.019145, nhưng vẫn cao hơn baseline 0.000227. Selective SSM thêm VN-Index có chênh lệch RMSE +0.000011 so với bản cơ sở; trong đó dấu dương nghĩa là RMSE tăng. DA của các mạng gần baseline train mean, nên không có cơ sở dùng DA để đảo ngược kết luận theo sai số.

Do đó, kết quả chính là một **kết quả âm có kiểm soát**: với feature hiện có và cửa sổ 30 phiên, mạng phức tạp chưa tạo lợi ích dự báo trung bình nhóm rõ ràng hơn baseline hằng. Đây là kết luận tốt hơn việc chỉ chọn vài ticker/seed đẹp hoặc chỉ báo cáo metric thuận lợi.

## 7. Thảo luận: vì sao kết quả có thể như vậy

Thứ nhất, return một phiên phụ thuộc nhiều vào thông tin mới, tâm lý và sự kiện chưa có trong OHLCV quá khứ. Mô hình chỉ thấy giá, khối lượng và một biến thị trường chung, nên không có lý do để mặc định nó vượt dự báo hằng. Thứ hai, quan hệ thị trường có thể thay đổi giữa giai đoạn train, validation và test; mẫu hình học được ở quá khứ không nhất thiết ổn định ở tương lai. Thứ ba, VN-Index là chỉ báo rộng và chỉ được đưa vào dưới dạng một return ngày; nó có thể chưa bổ sung thông tin ngoài feature giá của từng cổ phiếu.

Các giải thích này là giả thuyết hợp lý, không phải chứng minh nhân quả. Không thể kết luận Selective SSM không phù hợp với mọi dữ liệu tài chính; chỉ có thể kết luận cấu hình đã chạy chưa vượt baseline. Điều này cũng giải thích vì sao báo cáo giữ nguyên R2 âm và không biến DA thành kết luận đầu tư.

## 8. Độ tin cậy, khả năng tái lập và giới hạn

`EXPERIMENT_MANIFEST.json` lưu commit, SHA-256 các tệp mã chính, phiên bản Python/PyTorch, CPU/GPU, seed, window, epoch, batch size và manifest dữ liệu. Bộ chạy tạo 136 cấu hình và 59.704 dự báo; `verify_project.py` kiểm tra đủ lần chạy, tính hữu hạn và ghép đúng metric–prediction. Test đơn vị kiểm tra dữ liệu, quan hệ input–target, scaler không nhận giá trị tương lai, baseline, selective scan và metric.

Giới hạn quan trọng gồm: chỉ một khoảng test hậu 2025; chưa có walk-forward evaluation; chưa có news/sentiment, vĩ mô, báo cáo tài chính hay feature ngành; chưa tối ưu siêu tham số có kiểm soát; chưa có backtest chi phí giao dịch. Cảnh báo dữ liệu VN-Index được giữ lại trong manifest thay vì âm thầm sửa dữ liệu. Các giới hạn này cần xuất hiện trong báo cáo Word sau này.

## 9. Kết luận và hướng tiếp theo

Đề tài đã tạo một pipeline hồi quy return có kiểm soát leakage, có baseline, lặp nhiều seed, kiểm tra chất lượng dữ liệu và đầu ra tái lập. Tuy nhiên, câu trả lời hiện tại cho hai câu hỏi nghiên cứu là: chưa có bằng chứng Selective SSM vượt baseline train mean; thêm return VN-Index cũng chưa giảm RMSE. Hướng hợp lý là walk-forward validation, window 60/90 đã đăng ký trước, feature tin tức/sentiment hoặc vĩ mô, rồi mới mở rộng sang volatility forecast và risk classification bằng target riêng.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
3. Shi, Z. (2024). MambaStock: Selective State Space Model for Stock Prediction. arXiv:2402.18959.
4. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
5. Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica, 55(3), 703-708.

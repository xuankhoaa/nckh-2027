# Công thức và đề cương báo cáo NCKH

## 1. Phạm vi của đề tài hiện tại

Đề tài giải quyết **Bài toán 1: dự báo tỷ suất sinh lợi của cổ phiếu**. Cụ thể, mô hình dự báo tỷ suất sinh lợi logarit của **phiên kế tiếp** cho từng mã ngân hàng Việt Nam, dựa trên cửa sổ 30 phiên giao dịch gần nhất.

- Mã cổ phiếu: VCB, BID, CTG, MBB, TCB, ACB, VPB và STB.
- Khoảng dữ liệu thực nghiệm: từ 02/01/2020 đến 12/10/2026 (tùy mã có thể chênh lệch rất nhỏ do ngày niêm yết/giao dịch).
- Dữ liệu đầu vào gốc: giá mở cửa, cao nhất, thấp nhất, đóng cửa, khối lượng giao dịch (OHLCV) và chỉ số VN-Index.
- Đầu ra: một số thực liên tục là log-return của ngày $t+1$.

Đề tài **chưa** xây dựng nhãn độ biến động tương lai hoặc nhãn rủi ro thấp/trung bình/cao; vì vậy không nên trình bày kết quả như thể đã giải quyết Bài toán 2 hoặc Bài toán 3.

## 2. Ký hiệu sử dụng

| Ký hiệu | Ý nghĩa |
|---|---|
| $t$ | Phiên giao dịch hiện tại |
| $P_t$ | Giá đóng cửa của cổ phiếu tại phiên $t$ |
| $O_t,H_t,L_t$ | Giá mở cửa, cao nhất và thấp nhất tại phiên $t$ |
| $V_t$ | Khối lượng giao dịch tại phiên $t$ |
| $I_t$ | Giá đóng cửa VN-Index tại phiên $t$ |
| $\mathbf{x}_t$ | Vector đặc trưng tại phiên $t$ |
| $W$ | Độ dài cửa sổ thời gian; trong đề tài là $W=30$ |
| $y_t$ | Nhãn cần dự báo từ thông tin đến phiên $t$ |
| $\hat y_t$ | Giá trị dự báo của mô hình |
| $N$ | Số quan sát trong tập đánh giá |

## 3. Xây dựng đặc trưng và biến mục tiêu

### 3.1. Tỷ suất sinh lợi logarit

Tỷ suất sinh lợi logarit của cổ phiếu tại phiên $t$ được tính bởi:

$$
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right).
$$

Log-return được dùng thay cho chênh lệch giá tuyệt đối vì nó chuẩn hóa biến động theo quy mô giá và có tính cộng dồn theo thời gian:

$$
\ln\left(\frac{P_t}{P_{t-k}}\right)=\sum_{j=t-k+1}^{t} r_j.
$$

### 3.2. Thay đổi khối lượng giao dịch

$$
v_t=\frac{V_t}{V_{t-1}}-1.
$$

Đặc trưng này phản ánh mức tăng hoặc giảm tương đối của thanh khoản so với phiên trước.

### 3.3. Biên độ dao động trong ngày

$$
q_t=\frac{H_t-L_t}{P_t}.
$$

Chỉ số $q_t$ phản ánh mức dao động nội phiên, được chuẩn hóa theo giá đóng cửa.

### 3.4. Tỷ suất sinh lợi của VN-Index

$$
r_t^{(VN)}=\ln\left(\frac{I_t}{I_{t-1}}\right).
$$

Đây là đặc trưng thị trường chung. Nếu $r_t^{(VN)}>0$, VN-Index tăng so với phiên trước; nếu nhỏ hơn 0, thị trường chung giảm. Nó không phải là lợi nhuận của cổ phiếu đang dự báo.

### 3.5. Biến mục tiêu dự báo một phiên kế tiếp

Với thông tin chỉ có đến cuối phiên $t$, nhãn của mẫu là:

$$
y_t=r_{t+1}=\ln\left(\frac{P_{t+1}}{P_t}\right).
$$

Do đó, mô hình không được sử dụng bất kỳ thông tin nào của phiên $t+1$ trong đầu vào. Đây là nguyên tắc tránh rò rỉ dữ liệu tương lai.

### 3.6. Vector đặc trưng

Mô hình **Mamba cơ sở** và **LSTM** dùng vector:

$$
\mathbf{x}^{(base)}_t=[O_t,H_t,L_t,P_t,V_t,r_t,v_t,q_t].
$$

Mô hình **Improved Mamba** thêm thông tin thị trường chung:

$$
\mathbf{x}^{(imp)}_t=[O_t,H_t,L_t,P_t,V_t,r_t,v_t,q_t,r_t^{(VN)}].
$$

Lưu ý: mô hình LSTM là mô hình đối chứng (baseline) để so sánh; không phải là đặc trưng riêng.

## 4. Tạo chuỗi đầu vào bằng cửa sổ trượt

Từ các vector đặc trưng theo thời gian, mỗi mẫu được tạo thành:

$$
\mathbf{X}_t=[\mathbf{x}_{t-W+1},\mathbf{x}_{t-W+2},\ldots,\mathbf{x}_t], \quad W=30.
$$

Mô hình học ánh xạ:

$$
f_\theta:\mathbb{R}^{W\times d}\rightarrow\mathbb{R},\qquad
\hat y_t=f_\theta(\mathbf{X}_t),
$$

trong đó $d$ là số đặc trưng: $d=8$ với Mamba cơ sở/LSTM và $d=9$ với Improved Mamba.

Diễn giải: chuỗi gồm 30 phiên từ $t-29$ đến $t$ được dùng để dự báo return của phiên $t+1$.

## 5. Chia dữ liệu và chuẩn hóa

Mẫu được chia theo **thứ tự thời gian**, không xáo trộn:

| Tập dữ liệu | Thời gian theo ngày của nhãn $y_t$ |
|---|---|
| Huấn luyện | đến hết 31/12/2023 |
| Xác thực | từ 01/01/2024 đến 31/12/2024 |
| Kiểm tra | từ 01/01/2025 trở đi |

Việc chia theo ngày của nhãn giúp đảm bảo dự báo kiểm tra luôn nằm sau toàn bộ giai đoạn huấn luyện. Tập xác thực được dùng để chọn epoch dừng sớm; tập kiểm tra chỉ dùng một lần để báo cáo kết quả cuối cùng.

Mỗi đặc trưng được chuẩn hóa bằng StandardScaler, chỉ khớp trên phần dữ liệu đầu vào thuộc tập huấn luyện:

$$
z_{t,j}=\frac{x_{t,j}-\mu_j^{(train)}}{\sigma_j^{(train)}}.
$$

Ở đây, $\mu_j^{(train)}$ và $\sigma_j^{(train)}$ lần lượt là trung bình và độ lệch chuẩn của đặc trưng thứ $j$ trong dữ liệu huấn luyện. Tập xác thực và kiểm tra chỉ dùng các tham số này để biến đổi, không được tự tính lại.

## 6. Mô hình LSTM đối chứng

LSTM xử lý tuần tự qua các cổng quên, cổng vào và cổng ra. Với đầu vào $\mathbf{x}_t$, trạng thái ẩn $\mathbf{h}_{t-1}$ và trạng thái ô nhớ $\mathbf{c}_{t-1}$:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f),\\
\mathbf{i}_t &= \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i),\\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c),\\
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o),\\
\mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t).
\end{aligned}
$$

Sau trạng thái cuối của cửa sổ, một lớp tuyến tính cho dự báo:

$$
\hat y_t=\mathbf{w}^\top\mathbf{h}_t+b.
$$

## 7. Mô hình Mamba (Selective State Space Model)

### 7.1. Mô hình không gian trạng thái

Dạng liên tục tổng quát của State Space Model (SSM) là:

$$
\frac{d\mathbf{h}(\tau)}{d\tau}=\mathbf{A}\mathbf{h}(\tau)+\mathbf{B}\mathbf{x}(\tau),
\qquad
\mathbf{z}(\tau)=\mathbf{C}\mathbf{h}(\tau)+\mathbf{D}\mathbf{x}(\tau).
$$

Trong đó $\mathbf{h}$ là trạng thái ẩn, còn $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ là các tham số mô hình.

Sau rời rạc hóa theo thời gian, Mamba cập nhật theo dạng:

$$
\mathbf{h}_t=\bar{\mathbf{A}}_t\mathbf{h}_{t-1}+\bar{\mathbf{B}}_t\mathbf{x}_t,
\qquad
\mathbf{z}_t=\mathbf{C}_t\mathbf{h}_t+\mathbf{D}\mathbf{x}_t.
$$

### 7.2. Tính chọn lọc của Mamba

Khác với SSM cố định, Mamba làm các tham số phụ thuộc đầu vào:

$$
\Delta_t=s_\Delta(\mathbf{x}_t),\qquad
\mathbf{B}_t=s_B(\mathbf{x}_t),\qquad
\mathbf{C}_t=s_C(\mathbf{x}_t).
$$

Với bước thời gian $\Delta_t$, một dạng rời rạc hóa thường dùng là:

$$
\bar{\mathbf{A}}_t=\exp(\Delta_t\mathbf{A}).
$$

Trong phép xấp xỉ sử dụng trong lõi mã nguồn, hạng đưa đầu vào vào trạng thái có dạng $\bar{\mathbf{B}}_t\approx\Delta_t\mathbf{B}_t$. Do đó, cả bước thời gian, ma trận vào và ma trận đọc trạng thái đều có thể thay đổi theo từng phiên.

Nhờ đó, mô hình có thể điều chỉnh cách lưu giữ hoặc quên thông tin theo trạng thái thị trường tại từng phiên. Mã nguồn đề tài dùng lõi Mamba gốc, kết hợp chiếu đặc trưng đầu vào, tích chập một chiều theo thời gian, hàm kích hoạt SiLU, cơ chế cổng và kết nối phần dư; đầu ra cuối cửa sổ được đưa qua lớp hồi quy tuyến tính.

Về mặt thực nghiệm, Mamba được thiết kế để xử lý chuỗi với độ phức tạp tăng tuyến tính theo chiều dài chuỗi thông qua selective scan [Gu & Dao, 2023].

## 8. Huấn luyện mô hình

Hàm mất mát được tối thiểu hóa là sai số bình phương trung bình:

$$
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2.
$$

Các cấu hình thực nghiệm đang sử dụng:

| Thành phần | Cấu hình |
|---|---|
| Độ dài cửa sổ | 30 phiên |
| Số lớp mạng | 2 |
| Kích thước trạng thái ẩn | 16 |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Batch size | 128 |
| Số epoch tối đa | 75 |
| Early stopping | patience = 12, theo MSE xác thực |
| Hạt giống ngẫu nhiên | 42 |

Các tham số mặc định của lõi Mamba đang dùng là $d_{state}=16$, hệ số mở rộng $E=2$, tích chập thời gian một chiều có kernel $d_{conv}=4$ và selective scan song song. Với hidden size $D=16$, mô hình có kích thước biểu diễn bên trong là $E\times D=32$.

Early stopping giữ lại checkpoint có MSE xác thực tốt nhất và dừng khi không cải thiện sau 12 epoch liên tiếp. Điều này giúp giảm nguy cơ quá khớp.

## 9. Thước đo đánh giá

### 9.1. MSE và RMSE

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2,
\qquad
RMSE=\sqrt{MSE}.
$$

RMSE càng nhỏ càng tốt và có cùng đơn vị với log-return.

### 9.2. MAE

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|.
$$

MAE đo độ lệch tuyệt đối trung bình và ít nhạy với sai số rất lớn hơn RMSE.

### 9.3. Hệ số xác định \(R^2\)

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat y_i)^2}
{\sum_{i=1}^{N}(y_i-\bar y)^2}.
$$

Trong đó $\bar y$ là trung bình của giá trị thật trên tập kiểm tra. $R^2$ càng lớn càng tốt. Giá trị âm nghĩa là trên tập đó mô hình còn kém hơn phương án dự báo hằng bằng trung bình mẫu.

### 9.4. Độ chính xác chiều biến động (Directional Accuracy)

$$
DA=\frac{1}{N}\sum_{i=1}^{N}
\mathbb{I}\left[\operatorname{sign}(y_i)=\operatorname{sign}(\hat y_i)\right].
$$

Trong đó $\mathbb{I}[\cdot]$ là hàm chỉ báo. DA cho biết tỷ lệ mô hình dự báo đúng chiều tăng/giảm của return, nhưng không phản ánh độ lớn sai số; vì vậy cần báo cáo cùng RMSE, MAE và $R^2$.

## 10. Thuật toán thực nghiệm có thể đưa vào báo cáo

**Thuật toán 1. Dự báo log-return một phiên kế tiếp bằng LSTM/Mamba**

**Đầu vào:** Dữ liệu OHLCV từng mã ngân hàng, VN-Index, độ dài cửa sổ $W=30$.  
**Đầu ra:** $\hat y_t$, các chỉ số RMSE, MAE, $R^2$, DA trên tập kiểm tra.

1. Sắp xếp dữ liệu theo ngày tăng dần và loại các quan sát thiếu cần thiết.
2. Tính $r_t$, $v_t$, $q_t$ và $r_t^{(VN)}$.
3. Dịch $r_t$ lùi một phiên để tạo nhãn $y_t=r_{t+1}$.
4. Chia mẫu theo thời gian thành train, validation và test.
5. Fit StandardScaler chỉ bằng dữ liệu đầu vào train; biến đổi validation/test bằng cùng scaler.
6. Dùng cửa sổ trượt để tạo $(\mathbf{X}_t,y_t)$, với 30 phiên quá khứ dự báo một phiên kế tiếp.
7. Huấn luyện lần lượt LSTM, Mamba cơ sở và Improved Mamba bằng MSE/Adam; dùng validation để early stopping.
8. Khôi phục checkpoint tốt nhất, dự báo trên test và tính RMSE, MAE, $R^2$, DA.
9. Tổng hợp kết quả theo từng mã và theo từng nhóm mô hình.

## 11. Kết quả hiện có và cách diễn giải

Trung bình trên 8 mã ngân hàng ở tập kiểm tra:

| Mô hình | RMSE | MAE | $R^2$ | DA |
|---|---:|---:|---:|---:|
| LSTM | 0,026417 | 0,020189 | -1,078250 | 0,448747 |
| Mamba cơ sở | 0,053051 | 0,044755 | -7,573619 | **0,470387** |
| Improved Mamba | **0,024680** | **0,018623** | **-0,732823** | 0,446469 |

Kết luận phù hợp với số liệu là: **Improved Mamba tốt nhất theo các thước đo sai số hồi quy RMSE, MAE và $R^2$**; Mamba cơ sở có DA cao nhất nhưng sai số độ lớn lớn hơn nhiều. Không nên kết luận mô hình dự báo tốt về mặt thực tiễn khi $R^2$ trung bình vẫn âm; điều này phản ánh dự báo return ngắn hạn là bài toán nhiễu cao và là điểm cần thảo luận trung thực trong báo cáo.

Các bảng và biểu đồ đã sinh có thể chèn vào Chương Kết quả:

- `results/BANG_KET_QUA_CHI_TIET_VI.csv`: chỉ số theo từng mã và mô hình.
- `results/BANG_TONG_HOP_NHOM_VI.csv`: bảng trung bình theo mô hình.
- `results/DU_BAO_THEO_NGAY_VI.csv`: giá trị thật và dự báo theo ngày.
- `results/bieu_do/tung_ma/`: biểu đồ từng mã.
- `results/bieu_do/tong_hop/`: biểu đồ tổng hợp.

## 12. Đề cương báo cáo NCKH khuyến nghị

Nếu trường chưa có mẫu bắt buộc, có thể dùng cấu trúc sau. Nhiều hướng dẫn trình bày báo cáo khoa học dùng mạch: tóm tắt, từ khóa, giới thiệu, phương pháp, kết quả và thảo luận, kết luận, tài liệu tham khảo; yêu cầu riêng của trường luôn được ưu tiên hơn đề cương này.

1. **Trang bìa, trang phụ bìa, lời cam đoan, lời cảm ơn, mục lục, danh mục bảng và hình.**
2. **Tóm tắt** (khoảng 200–300 từ): vấn đề, dữ liệu, ba mô hình, kết quả chính và đóng góp. Thêm từ khóa: *dự báo tỷ suất sinh lợi; cổ phiếu ngân hàng; Mamba; LSTM; VN-Index*.
3. **Chương 1 — Mở đầu**
   - Lý do chọn đề tài và tính cấp thiết.
   - Mục tiêu tổng quát, mục tiêu cụ thể.
   - Câu hỏi nghiên cứu: Mamba có giảm sai số dự báo return ngày kế tiếp so với LSTM không? Việc thêm return VN-Index có cải thiện kết quả không?
   - Đối tượng, phạm vi: 8 mã ngân hàng; dữ liệu 2020–2026; dự báo 1 phiên kế tiếp.
   - Đóng góp: pipeline tránh rò rỉ dữ liệu, so sánh ba mô hình, kết quả thực nghiệm trên ngân hàng Việt Nam.
4. **Chương 2 — Tổng quan nghiên cứu và cơ sở lý thuyết**
   - Tổng quan dự báo chuỗi thời gian tài chính.
   - LSTM và các nghiên cứu liên quan.
   - State Space Model, Mamba và MambaStock.
   - Khoảng trống nghiên cứu: áp dụng/đánh giá Mamba trên nhóm cổ phiếu ngân hàng Việt Nam với quy trình chia thời gian rõ ràng.
5. **Chương 3 — Dữ liệu và phương pháp nghiên cứu**
   - Nguồn dữ liệu, phạm vi thời gian, danh sách mã.
   - Làm sạch, feature dẫn xuất và các công thức ở Mục 3.
   - Tạo target, cửa sổ trượt, chuẩn hóa và chia train/validation/test.
   - Kiến trúc LSTM, Mamba cơ sở, Improved Mamba.
   - Thiết lập thực nghiệm, phần cứng/phần mềm, siêu tham số và thước đo.
6. **Chương 4 — Kết quả và thảo luận**
   - Thống kê mô tả dữ liệu.
   - Bảng kết quả tổng hợp và bảng theo từng mã.
   - Biểu đồ actual–predicted/loss; phân tích trường hợp Mamba tốt hoặc chưa tốt.
   - So sánh công bằng giữa ba mô hình; thảo luận $R^2$ âm, nhiễu thị trường, giới hạn chưa dùng tin tức/sentiment.
7. **Kết luận và kiến nghị**
   - Trả lời từng mục tiêu/câu hỏi nghiên cứu bằng số liệu.
   - Nêu giới hạn; đề xuất mở rộng Bài toán 2 (volatility) hoặc Bài toán 3 (risk classification) ở nghiên cứu sau.
8. **Tài liệu tham khảo**: dùng một chuẩn thống nhất (APA 7 hoặc IEEE theo quy định trường).
9. **Phụ lục**: bảng mã, tham số đầy đủ, mô tả cấu trúc thư mục, hướng dẫn chạy. Không cần dán toàn bộ mã nguồn vào thân báo cáo.

Với dạng bài báo ngắn/hội nghị, rút gọn thành: **Tiêu đề – Tóm tắt – Từ khóa – 1. Giới thiệu – 2. Phương pháp – 3. Kết quả và thảo luận – 4. Kết luận – Tài liệu tham khảo**.

## 13. Tài liệu tham khảo khởi đầu

Các mục dưới đây là tài liệu nền tảng; khi nộp chính thức cần trình bày lại theo chuẩn trích dẫn của trường.

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. Hướng dẫn trình bày báo cáo Hội nghị NCKH quốc gia về Nghiên cứu và Giảng dạy Sinh học ở Việt Nam lần thứ 7. Trường Đại học Sư phạm – Đại học Thái Nguyên. https://bio.tnue.edu.vn/tin-hoat-dong/huong-dan-ban-thao-va-trinh-bay-bao-cao-khoa-hoc-tai-hoi-nghi-khoa-hoc-quoc-gia-ve-nghien-cuu-va-giang-day-sinh-hoc-o-viet-nam-lan-thu-7-2026
5. Sổ tay công tác nghiên cứu khoa học sinh viên. Trường Đại học Nông Lâm – Đại học Thái Nguyên. https://tuaf.edu.vn/images/docs/2021/06/07/20210607095440so-tay-nckh-pdf.pdf

## 14. Lưu ý khi viết để báo cáo đáng tin cậy

- Chỉ mô tả những gì đã chạy: không ghi sentiment, volatility forecast hay phân loại rủi ro như một thành phần đã có.
- Phân biệt rõ **return của cổ phiếu** $r_t$ với **return VN-Index** $r_t^{(VN)}$.
- Báo cáo cả chỉ số tốt lẫn hạn chế. Không che $R^2$ âm hoặc chỉ chọn DA để kết luận.
- Khi so sánh, dùng cùng dữ liệu, cùng cách chia thời gian, cùng cửa sổ 30 phiên và cùng thước đo cho cả ba mô hình.
- Nếu trường cung cấp mẫu bìa, định dạng trang hoặc mẫu biểu riêng, mẫu đó có ưu tiên cao hơn đề cương này.

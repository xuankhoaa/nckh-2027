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

Nhờ đó, mô hình có thể điều chỉnh cách lưu giữ hoặc quên thông tin theo trạng thái thị trường tại từng phiên. Mã nguồn đề tài **tự triển khai** một Selective SSM tối giản từ các phương trình trên, gồm chiếu đặc trưng đầu vào, tích chập một chiều nhân quả, hàm SiLU, cơ chế cổng, chuẩn hóa RMS và kết nối phần dư; đầu ra tại thời điểm cuối cửa sổ đi qua lớp hồi quy tuyến tính. Kho mã không sao chép hoặc vendoring mã nguồn MambaStock.

Về mặt thực nghiệm, Mamba được thiết kế để xử lý chuỗi với độ phức tạp tăng tuyến tính theo chiều dài chuỗi thông qua selective scan [Gu & Dao, 2023].

### 7.3. Quét trạng thái trong hiện thực đề tài

Với mỗi thành phần trạng thái, mã sử dụng truy hồi rời rạc:

$$
h_t=a_t\odot h_{t-1}+b_t,\qquad h_{-1}=0,
$$

trong đó $a_t=\exp(\Delta_t A)$ và $b_t$ là hạng đưa input có phụ thuộc $\Delta_t$, $B_t$ và input đã qua causal convolution. Hai phép biến đổi affine liên tiếp được ghép theo:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2\odot a_1,\ b_2+a_2\odot b_1).
$$

Phép ghép này có tính kết hợp. Do đó, chương trình dùng prefix scan theo cây để nhận toàn bộ $h_t$ với $\lceil\log_2 W\rceil$ bước tensor thay vì vòng lặp Python theo từng phiên. Đây là tối ưu hiện thực; test đơn vị xác nhận kết quả bằng truy hồi tuần tự.

## 8. Huấn luyện mô hình

Hàm mất mát được tối thiểu hóa là sai số bình phương trung bình trên nhãn đã chuẩn hóa theo train:

$$
\tilde y=\frac{y-\mu_{train}}{\sigma_{train}},
\qquad
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\tilde y_i-\widehat{\tilde y}_i)^2.
$$

Trong đó $\mu_{train}$ và $\sigma_{train}$ chỉ được tính từ $y_{train}$. Khi đánh giá, dự báo được đảo chuẩn hóa:

$$
\hat y=\widehat{\tilde y}\sigma_{train}+\mu_{train}.
$$

Các cấu hình thực nghiệm đang sử dụng:

| Thành phần | Cấu hình |
|---|---|
| Độ dài cửa sổ | 30 phiên |
| Số lớp mạng | 1 |
| Kích thước trạng thái ẩn | 8 |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Batch size | 128 |
| Số epoch tối đa | 40 |
| Early stopping | patience = 8, theo MSE xác thực |
| Hạt giống ngẫu nhiên | 42, 52, 62, 72, 82 |

Các tham số mặc định của Selective SSM là $d_{state}=8$, hệ số mở rộng $E=1$, tích chập thời gian nhân quả có kernel $d_{conv}=3$. Với hidden size $D=8$, mô hình có kích thước biểu diễn bên trong $E\times D=8$. LSTM có 585 tham số, Selective SSM cơ sở có 545 tham số và bản thêm VN-Index có 553 tham số; các quy mô này gần nhau để so sánh công bằng trên bộ dữ liệu nhỏ.

Early stopping giữ lại checkpoint có MSE xác thực tốt nhất và dừng khi không cải thiện sau 8 epoch liên tiếp. Hai baseline `naive_zero` và `train_mean` được thêm vào để tránh việc so sánh mạng chỉ với nhau.

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

1. Sắp xếp dữ liệu theo ngày tăng dần; kiểm tra schema, giá trị thiếu, điều kiện OHLCV và lưu SHA-256 của từng tệp.
2. Tính $r_t$, $v_t$, $q_t$ và $r_t^{(VN)}$; tạo nhãn $y_t=r_{t+1}$ cùng ngày nhãn `target_date`.
3. Fit StandardScaler chỉ bằng feature có trong train; tạo cửa sổ $(\mathbf{X}_t,y_t)$ dài 30 phiên và chia mẫu theo `target_date` thành train, validation, test.
4. Chuẩn hóa nhãn bằng $\mu_{train},\sigma_{train}$; không fit lại trên validation hoặc test.
5. Với mỗi mã, tạo dự báo `naive_zero` và `train_mean`; huấn luyện LSTM, Selective SSM cơ sở, Selective SSM có VN-Index trên năm seed.
6. Dùng MSE/Adam, validation để early stopping; khôi phục checkpoint tốt nhất và đảo chuẩn hóa output.
7. Dự báo trên test và tính RMSE, MAE, $R^2$, DA; kiểm định chênh lệch squared loss có phương sai HAC so với baseline zero.
8. Tổng hợp mean và độ lệch chuẩn qua seed, sau đó qua các mã; lưu manifest môi trường và báo cáo tự động.

## 11. Kết quả hiện có và cách diễn giải

Không chép tay metric vào tài liệu này. Sau khi chạy đủ 136 cấu hình, `generate_report.py` sinh trực tiếp các bảng và phần diễn giải từ `results/metrics.csv`. Vì vậy, số liệu trong báo cáo luôn có thể truy vết về seed, ticker, feature set, commit mã và SHA-256 dữ liệu.

Khi viết phần kết quả, dùng `BANG_TONG_HOP_NHOM_VI.csv` để báo cáo trung bình và độ lệch chuẩn. Mô hình có RMSE/MAE thấp nhất chỉ là lựa chọn tốt nhất **trong snapshot và thiết lập đã kiểm tra**. Không chọn mô hình chỉ theo DA; nếu $R^2$ âm, phải nêu rõ rằng mô hình kém hơn dự báo hằng bằng trung bình test theo chỉ số đó. `KIEM_DINH_HAC_VI.csv` là bằng chứng bổ sung về squared loss, không phải lý do duy nhất để kết luận.

Các bảng và biểu đồ đã sinh có thể chèn vào Chương Kết quả:

- `results/BANG_KET_QUA_CHI_TIET_VI.csv`: chỉ số theo từng mã và mô hình.
- `results/BANG_TONG_HOP_NHOM_VI.csv`: bảng trung bình theo mô hình.
- `results/DU_BAO_THEO_NGAY_VI.csv`: giá trị thật và dự báo theo ngày.
- `results/bieu_do/tung_ma/`: biểu đồ từng mã.
- `results/bieu_do/tong_hop/`: biểu đồ tổng hợp.

## 12. Tài liệu tham khảo khởi đầu

Các mục dưới đây là tài liệu nền tảng; khi nộp chính thức cần trình bày lại theo chuẩn trích dẫn của trường.

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. Hướng dẫn trình bày báo cáo Hội nghị NCKH quốc gia về Nghiên cứu và Giảng dạy Sinh học ở Việt Nam lần thứ 7. Trường Đại học Sư phạm – Đại học Thái Nguyên. https://bio.tnue.edu.vn/tin-hoat-dong/huong-dan-ban-thao-va-trinh-bay-bao-cao-khoa-hoc-tai-hoi-nghi-khoa-hoc-quoc-gia-ve-nghien-cuu-va-giang-day-sinh-hoc-o-viet-nam-lan-thu-7-2026
5. Sổ tay công tác nghiên cứu khoa học sinh viên. Trường Đại học Nông Lâm – Đại học Thái Nguyên. https://tuaf.edu.vn/images/docs/2021/06/07/20210607095440so-tay-nckh-pdf.pdf

## 13. Lưu ý khi viết để báo cáo đáng tin cậy

- Chỉ mô tả những gì đã chạy: không ghi sentiment, volatility forecast hay phân loại rủi ro như một thành phần đã có.
- Phân biệt rõ **return của cổ phiếu** $r_t$ với **return VN-Index** $r_t^{(VN)}$.
- Báo cáo cả chỉ số tốt lẫn hạn chế. Không che $R^2$ âm hoặc chỉ chọn DA để kết luận.
- Khi so sánh, dùng cùng dữ liệu, cùng cách chia thời gian, cùng cửa sổ 30 phiên và cùng thước đo cho mọi mô hình.
- Nếu trường cung cấp mẫu bìa, định dạng trang hoặc mẫu biểu riêng, hãy tuân thủ đúng mẫu đó.

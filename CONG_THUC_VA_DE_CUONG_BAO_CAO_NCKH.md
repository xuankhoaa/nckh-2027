# Đặc tả phương pháp và công thức nghiên cứu

## 1. Ký hiệu

- $P_t,O_t,H_t,L_t$: giá đóng/mở/cao/thấp của cổ phiếu tại phiên $t$.
- $V_t$: khối lượng.
- $I_t$: giá đóng cửa VN-Index.
- $s_t$: sentiment trung bình của tin được phép dùng tại phiên $t$.
- $n_t$: số tin được phép dùng tại phiên $t$.
- $W$: chiều dài cửa sổ input, mặc định 30.
- $H$: horizon volatility, mặc định 5.

## 2. Feature thị trường

$$
r_t=\ln(P_t/P_{t-1}),\quad
\Delta V_t=V_t/V_{t-1}-1,
$$

$$
q_t=(H_t-L_t)/P_t,\quad
r_t^{VN}=\ln(I_t/I_{t-1}),
$$

$$
\widehat\sigma_{t,5}=\sqrt{\frac{1}{5}\sum_{j=0}^{4}r_{t-j}^2}.
$$

Market-only gồm OHLCV và các đại lượng trên. Market+News bổ sung $(s_t,n_t)$.

## 3. Đồng bộ tin tức

Gọi $c$ là cutoff theo giờ Việt Nam. Ngày khả dụng sơ bộ của tin $i$ là:

$$
d_i^{avail}=\begin{cases}
d_i,& hour_i<c,\\
d_i+1,& hour_i\ge c.
\end{cases}
$$

Tin được ánh xạ sang phiên giao dịch đầu tiên không sớm hơn $d_i^{avail}$. Vì vậy tin sau đóng cửa, cuối tuần hoặc ngày nghỉ không đi ngược về một phiên trước khi nó xuất hiện. Trên mỗi mã và phiên:

$$
s_t=\frac{1}{n_t}\sum_{i\in\mathcal N_t}s_i,
\qquad n_t=|\mathcal N_t|.
$$

Nếu $\mathcal N_t=\varnothing$, đặt $(s_t,n_t)=(0,0)$.

## 4. Hai biến mục tiêu

### Return

$$
y_t^{(r)}=r_{t+1}.
$$

### Volatility

$$
y_t^{(\sigma)}=\sqrt{\frac{1}{H}\sum_{j=1}^{H}r_{t+j}^2}.
$$

Input là $X_t=[x_{t-W+1},\ldots,x_t]$. Ngày chia tập của return là $t+1$; ngày chia tập của volatility là $t+H$. Quy tắc này bảo đảm toàn bộ nhãn nằm trong đúng giai đoạn.

## 5. Chuẩn hóa

Với từng feature $k$, chỉ dùng train:

$$
z_{t,k}=\frac{x_{t,k}-\mu_k^{train}}{\sigma_k^{train}}.
$$

Target cũng được chuẩn hóa bằng mean/std train khi tối ưu và đảo chuẩn hóa trước đánh giá.

## 6. Loss

Cấu hình chính dùng MSE:

$$
\mathcal L=\frac1N\sum_i(\hat y_i-y_i)^2.
$$

Độ nhạy tail loss dùng ngưỡng phân vị 90% chỉ tính từ train:

$$
\mathcal L_w=\frac1N\sum_i w_i(\hat y_i-y_i)^2,
\qquad
w_i=\begin{cases}\lambda,& score(y_i)\ge Q_{0.9}^{train},\\1,&\text{khác.}\end{cases}
$$

Trong đó $score(y)=|y|$ cho return và $score(y)=y$ cho volatility; $\lambda\in\{1,2\}$ ở sensitivity.

## 7. Baseline và mô hình

Train Mean:

$$
\hat y_t=\bar y_{train}.
$$

Zero-return chỉ cho bài toán return:

$$
\hat y_t=0.
$$

LSTM và Selective SSM nhận tensor `[batch, W, features]`, dùng trạng thái cuối để hồi quy một giá trị. Selective SSM triển khai truy hồi chọn lọc:

$$
h_t=A_t\odot h_{t-1}+B_t\odot x_t,
\qquad
o_t=C_t\odot h_t+D\odot x_t,
$$

với $A_t,B_t,C_t$ phụ thuộc đầu vào và causal convolution không dùng tương lai. Đây là hiện thực Mamba-style tối giản, không phải benchmark chính thức của `mamba_ssm`.

## 8. Metric

$$
RMSE=\sqrt{\frac1N\sum_i(y_i-\hat y_i)^2},
\qquad
MAE=\frac1N\sum_i|y_i-\hat y_i|.
$$

Hệ số xác định thông thường trên test và $R^2_{OOS}$ so với Train Mean được lưu tách biệt:

$$
R^2_{test}=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y_{test})^2},
$$

$$
R^2_{OOS}=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\hat y_i^{TrainMean})^2}.
$$

Theo định nghĩa này, Train Mean có $R^2_{OOS}=0$; giá trị âm là kém Train Mean và giá trị dương là vượt Train Mean.

Directional Accuracy cho return:

$$
DA=\frac1N\sum_i\mathbf 1[sign(y_i)=sign(\hat y_i)].
$$

QLIKE cho volatility, dùng variance $v_i=y_i^2$ và $\hat v_i=\hat y_i^2$:

$$
QLIKE=\frac1N\sum_i\left(\frac{v_i}{\hat v_i}-\ln\frac{v_i}{\hat v_i}-1\right).
$$

## 9. So sánh đúng

### So sánh kiến trúc

$$
\Delta_{arch}=RMSE(Mamba,F)-RMSE(LSTM,F)
$$

với cùng tập feature $F$. Giá trị âm có lợi cho Mamba.

### Ablation dữ liệu

$$
\Delta_{news}=RMSE(M,Market+News)-RMSE(M,Market-only)
$$

với cùng mô hình $M$. Giá trị âm cho thấy news có lợi. Hai ablation thành phần thay `Market+News` lần lượt bằng `Market+news_count` và `Market+sentiment`.

Không so Mamba+News với LSTM Market-only để kết luận về kiến trúc hoặc về news, vì hai yếu tố thay đổi đồng thời.

## 10. Tổng hợp và kiểm định

Với mỗi mã/cấu hình, tính mean và sample standard deviation qua 5 seed. Sau đó lấy trung bình qua tám mã để mỗi mã có trọng số bằng nhau. Baseline xác định có độ lệch chuẩn seed bằng 0.

Loss differential so với Train Mean:

$$
d_t=(y_t-\hat y_t^{mean})^2-(y_t-\hat y_t^{model})^2.
$$

$\bar d>0$ có lợi cho mô hình. Sai số chuẩn dùng Newey–West/HAC; với volatility horizon chồng lấn, lag tối thiểu $H-1$.

## 11. Ma trận chính

| Target | Baseline | Mạng | Feature |
|---|---|---|---|
| Return | Zero, Train Mean | LSTM, Mamba | Market-only, Market+News |
| Volatility | Train Mean | LSTM, Mamba | Market-only, Market+News |

Mỗi mạng chạy 5 seed trên 8 mã. Hai ablation và sensitivity được báo cáo tách khỏi ma trận chính để tránh làm mờ giả thuyết chính.

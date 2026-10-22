# NCKH 2027 — Dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Mamba

## 1. Giới thiệu

Đây là mã nguồn phục vụ đề tài nghiên cứu khoa học về dự báo **tỷ suất sinh lợi logarit của phiên giao dịch kế tiếp** cho nhóm cổ phiếu ngân hàng Việt Nam. Đề tài so sánh mô hình LSTM với hai cấu hình Mamba trên cùng một quy trình dữ liệu theo thời gian, có kiểm soát rò rỉ dữ liệu tương lai.

**Câu hỏi nghiên cứu chính:** Với thông tin thị trường có đến cuối phiên $t$, mô hình nào dự báo tốt hơn tỷ suất sinh lợi logarit của phiên $t+1$? Việc bổ sung tỷ suất sinh lợi của VN-Index có giúp Mamba giảm sai số hay không?

> Đây là **bài toán hồi quy dự báo tỷ suất sinh lợi**. Kho mã hiện chưa xây dựng bài toán dự báo độ biến động tương lai, phân loại rủi ro thấp/trung bình/cao, hoặc đặc trưng sentiment từ tin tức.

Tài liệu công thức, thuật toán và đề cương báo cáo NCKH: [CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md](CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md).

Hướng dẫn thao tác từ đầu vào đến đầu ra: [HUONG_DAN_A_Z.md](HUONG_DAN_A_Z.md).

## 2. Phạm vi dữ liệu và đầu vào

| Nội dung | Mô tả |
|---|---|
| Nhóm cổ phiếu | VCB, BID, CTG, MBB, TCB, ACB, VPB và STB |
| Chỉ số thị trường | VN-Index |
| Dữ liệu gốc | Giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng giao dịch hằng ngày (OHLCV) |
| Khoảng dữ liệu đã chạy | 02/01/2020–12/10/2026; ACB có 1.684 phiên, các mã còn lại và VN-Index có 1.689 phiên |
| Đơn vị dự báo | Một phiên giao dịch kế tiếp |
| Độ dài cửa sổ đầu vào | 30 phiên giao dịch |

Dữ liệu được lấy bằng Vnstock, lưu cục bộ trong `data/` sau khi tải. Thư mục này không được đưa lên GitHub vì có thể tải lại bằng mã nguồn; các lần chạy sử dụng dữ liệu đã lưu sẽ không gọi lại nguồn dữ liệu nếu không có tùy chọn `--force`.

## 3. Bài toán và cách tạo dữ liệu

Gọi $P_t$ là giá đóng cửa tại phiên $t$. Tỷ suất sinh lợi logarit của cổ phiếu được tính như sau:

```text
return_t = ln(P_t / P_(t-1))
```

Nhãn cần dự báo của mỗi mẫu là tỷ suất sinh lợi ở phiên kế tiếp:

```text
target_return_t = return_(t+1) = ln(P_(t+1) / P_t)
```

Từ một cửa sổ 30 phiên kết thúc tại $t$, mô hình dự báo `target_return_t`. Nói cách khác, đầu vào chỉ chứa thông tin đã biết đến cuối phiên $t$; không chứa giá, khối lượng hay return của phiên $t+1$.

Các đặc trưng dẫn xuất được tính bằng:

| Đặc trưng | Công thức | Ý nghĩa |
|---|---|---|
| `return` | `ln(close_t / close_(t-1))` | Tỷ suất sinh lợi logarit của cổ phiếu |
| `volume_change` | `volume_t / volume_(t-1) - 1` | Thay đổi tương đối của thanh khoản |
| `high_low_range` | `(high_t - low_t) / close_t` | Biên độ dao động nội phiên đã chuẩn hóa |
| `vnindex_return` | `ln(VNIndex_t / VNIndex_(t-1))` | Tỷ suất sinh lợi logarit của thị trường chung |

`vnindex_return` là return của VN-Index, không phải return của cổ phiếu đang dự báo. Sau khi ghép dữ liệu cổ phiếu với VN-Index theo ngày giao dịch, chỉ các ngày có dữ liệu ở cả hai bên mới được sử dụng.

## 4. Các mô hình được so sánh

| Mô hình | Đầu vào | Vai trò trong thực nghiệm |
|---|---|---|
| LSTM | OHLCV và ba đặc trưng dẫn xuất `return`, `volume_change`, `high_low_range` | Mô hình đối chứng phổ biến cho chuỗi thời gian |
| Mamba cơ sở | Cùng tập đặc trưng với LSTM | Đánh giá lõi Mamba khi chưa có thông tin chỉ số thị trường |
| Improved Mamba | Tập đặc trưng của Mamba cơ sở cộng `vnindex_return` | Kiểm tra tác động của thông tin thị trường chung |

Mô hình Mamba tái sử dụng lõi Selective State Space Model từ [MambaStock](https://github.com/zshicode/MambaStock) trong `external/MambaStock/`. Lớp `models/mamba_model.py` bổ sung chiếu đầu vào và đầu ra hồi quy một giá trị tại trạng thái cuối của cửa sổ. Cấu hình thực nghiệm: 2 lớp, hidden size 16, trạng thái SSM 16, hệ số mở rộng 2 và tích chập thời gian một chiều có kernel 4.

## 5. Kiểm soát rò rỉ dữ liệu và thiết kế đánh giá

Chuỗi tài chính không được chia ngẫu nhiên. Mỗi mẫu được gán vào tập dữ liệu theo **ngày của nhãn cần dự báo**, không phải theo ngày bắt đầu cửa sổ.

| Tập | Khoảng thời gian của ngày nhãn |
|---|---|
| Huấn luyện | Đến hết 31/12/2023 |
| Xác thực | 01/01/2024–31/12/2024 |
| Kiểm tra | Từ 01/01/2025 đến 12/10/2026 |

Quy trình có năm biện pháp chống rò rỉ dữ liệu:

1. `target_return` và `target_date` được dịch sang phiên kế tiếp.
2. Mỗi chuỗi chỉ dùng 30 phiên kết thúc ở $t$ để dự báo $t+1$.
3. Dữ liệu được chia theo thứ tự thời gian; không dùng random split.
4. `StandardScaler` chỉ được fit bằng các quan sát đầu vào thuộc giai đoạn huấn luyện. Tập xác thực và kiểm tra chỉ dùng lại các tham số chuẩn hóa của tập huấn luyện.
5. Epoch tốt nhất được chọn theo MSE trên tập xác thực. Tập kiểm tra chỉ được dùng để đánh giá cuối cùng của một lần chạy.

## 6. Thiết lập huấn luyện và thước đo

Lần chạy toàn bộ nhóm dùng hạt giống ngẫu nhiên 42, Adam với `learning_rate = 0.001`, batch size 128, tối đa 75 epoch và early stopping sau 12 epoch không cải thiện MSE trên tập xác thực. Mô hình sẽ dùng GPU nếu PyTorch nhận diện được; nếu không, mô hình chạy trên CPU.

Các thước đo được báo cáo trên tập kiểm tra:

| Thước đo | Cách hiểu |
|---|---|
| RMSE | Căn bậc hai của trung bình bình phương sai số; càng thấp càng tốt |
| MAE | Trung bình sai số tuyệt đối; càng thấp càng tốt |
| R² | Mức độ giải thích biến thiên; càng lớn càng tốt. R² âm nghĩa là mô hình kém hơn dự báo hằng bằng trung bình tập kiểm tra trong cấu hình đó |
| Độ chính xác hướng (DA) | Tỷ lệ dự báo đúng dấu tăng/giảm của return; cần đọc cùng RMSE và MAE vì không phản ánh độ lớn sai số |

## 7. Kết quả của lần chạy đầy đủ hiện tại

Kết quả dưới đây là trung bình trên 8 mã cổ phiếu, với cửa sổ 30 phiên. Đây là kết quả đánh giá lịch sử đến 12/10/2026, không phải dự báo chắc chắn cho tương lai.

| Mô hình | RMSE trung bình | MAE trung bình | R² trung bình | DA trung bình |
|---|---:|---:|---:|---:|
| LSTM | 0,026417 | 0,020189 | -1,078250 | 0,448747 |
| Mamba cơ sở | 0,053051 | 0,044755 | -7,573619 | **0,470387** |
| Improved Mamba | **0,024680** | **0,018623** | **-0,732823** | 0,446469 |

**Diễn giải đúng theo số liệu:** Improved Mamba có sai số RMSE và MAE thấp nhất, đồng thời có R² trung bình cao nhất trong ba mô hình. Mamba cơ sở có DA cao nhất nhưng sai số độ lớn lại cao hơn đáng kể. Vì vậy không thể kết luận một mô hình tốt hơn hoàn toàn ở mọi tiêu chí. R² trung bình của cả ba mô hình đều âm; đây là hạn chế quan trọng, cho thấy dự báo return một phiên là bài toán có nhiễu cao và chưa đạt hiệu quả tốt hơn dự báo trung bình trên tập kiểm tra.

Bảng theo từng mã, dự báo theo ngày và biểu đồ được tạo tại `results/` sau khi chạy. Thư mục `results/` được Git bỏ qua vì là đầu ra tái tạo được.

## 8. Cấu trúc mã nguồn

| Tệp/thư mục | Chức năng |
|---|---|
| `config.py` | Danh sách 8 mã, tập đặc trưng và mốc chia thời gian |
| `get_data.py` | Tải, kiểm tra và lưu dữ liệu OHLCV theo ngày |
| `preprocess.py` | Tạo đặc trưng, nhãn, chuẩn hóa và cửa sổ trượt không rò rỉ dữ liệu |
| `models/lstm.py` | Mô hình LSTM đối chứng |
| `models/mamba_model.py` | Mô hình hồi quy dùng lõi Mamba |
| `external/MambaStock/` | Mã lõi Mamba và selective scan được sử dụng lại, có ghi nguồn |
| `train.py` | Huấn luyện một cấu hình, early stopping và lưu đánh giá |
| `run_experiments.py` | Chạy 3 mô hình cho toàn bộ 8 mã |
| `evaluate.py` | Tính metric, lưu bảng dự báo và vẽ biểu đồ |
| `generate_report.py` | Sinh bảng/báo cáo kết quả tiếng Việt từ đầu ra đã lưu |
| `CONG_THUC_VA_DE_CUONG_BAO_CAO_NCKH.md` | Công thức toán, thuật toán và đề cương viết báo cáo |
| `HUONG_DAN_A_Z.md` | Hướng dẫn sử dụng từ đầu vào đến đầu ra |

## 9. Cài đặt và chạy lại thí nghiệm

Mã thu thập dữ liệu hiện sử dụng API `Quote` của **Vnstock 4.x**. Môi trường kiểm chứng là Vnstock 4.0.8; không dùng Vnstock 5.x cho phiên bản mã này.

1. Chuẩn bị Python 3.10 trở lên và môi trường ảo phù hợp.
2. Tạo tệp `.env` từ `.env.example`, sau đó đặt khóa Vnstock của bạn vào biến `VNSTOCK_API_KEY`. Không đưa khóa vào mã nguồn hoặc GitHub.
3. Cài thư viện:

   ```powershell
   python -m pip install -U --extra-index-url https://vnstocks.com/api/simple -r requirements.txt
   ```

4. Tải dữ liệu cho toàn bộ nhóm ngân hàng và VN-Index:

   ```powershell
   python get_data.py --symbols VCB BID CTG MBB TCB ACB VPB STB
   ```

5. Chạy ba mô hình và sinh báo cáo tiếng Việt:

   ```powershell
   python run_experiments.py --window 30 --epochs 75 --batch-size 128
   python generate_report.py --window 30
   ```

Tệp đầu ra quan trọng:

- `results/BAO_CAO_KET_QUA.md`: báo cáo kết quả tổng hợp bằng tiếng Việt.
- `results/BANG_KET_QUA_CHI_TIET_VI.csv`: kết quả đầy đủ theo mã và mô hình.
- `results/BANG_TONG_HOP_NHOM_VI.csv`: trung bình và độ lệch chuẩn theo mô hình.
- `results/DU_BAO_THEO_NGAY_VI.csv`: return thực tế và return dự báo theo ngày kiểm tra.
- `results/bieu_do/`: biểu đồ loss, dự báo–thực tế và so sánh các metric.

Chỉ dùng `--force` khi thực sự muốn tải lại/ghi đè dữ liệu đã lưu trong `data/`.

## 10. Giới hạn và hướng mở rộng

- Đề tài chỉ dùng dữ liệu giá, khối lượng và return VN-Index; chưa sử dụng báo cáo tài chính, biến kinh tế vĩ mô hay sentiment từ tin tức.
- Kết quả mới là một thiết lập dữ liệu và siêu tham số; chưa có kiểm định thống kê hoặc tối ưu siêu tham số toàn diện.
- R² âm cho thấy không nên dùng trực tiếp kết quả như khuyến nghị mua/bán hay tư vấn đầu tư.
- Hướng mở rộng phù hợp là: tạo target độ biến động tương lai cho Bài toán 2; xây nhãn rủi ro thấp/trung bình/cao và đánh giá classification cho Bài toán 3; hoặc bổ sung sentiment và kiểm định qua nhiều cửa sổ thời gian.

## 11. Tài liệu tham khảo nền tảng

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

# Ghi chú nguồn tham khảo và mã bên thứ ba

Kho mã này không chứa mã nguồn sao chép từ MambaStock. Thành phần `models/mamba_model.py` là hiện thực độc lập, tối giản của Selective State Space Model phục vụ hồi quy; nó được viết từ các phương trình và mô tả thuật toán trong bài báo Mamba.

Các tài liệu học thuật được tham khảo:

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752
2. Shi, Z. (2024). *MambaStock: Selective State Space Model for Stock Prediction*. arXiv:2402.18959. https://arxiv.org/abs/2402.18959
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

`MambaStock` được nêu như một tham khảo học thuật; mã nguồn repository đó không được vendoring vào dự án. Thư viện Python được cài đặt khi chạy dự án tuân theo license riêng của từng gói; xem thông tin của các gói trong môi trường cài đặt trước khi phân phối sản phẩm.

## Dữ liệu bên thứ ba

- Dữ liệu OHLCV và VN-Index trong `data/` được truy xuất từ nguồn KBS thông qua
  công cụ Vnstock.
- Dữ liệu tin trong `data/` được truy hồi qua Google News RSS; trường `source`
  và `source_url` xác định nhà xuất bản gốc.
- Snapshot tin chỉ lưu metadata, tiêu đề, liên kết và các đặc trưng dẫn xuất;
  repository không chứa toàn văn bài báo.

Các tệp dữ liệu được công bố nhằm hỗ trợ kiểm tra và tái lập kết quả nghiên cứu.
Việc đưa dữ liệu vào repository không chuyển giao quyền sở hữu hoặc cấp thêm
giấy phép đối với nội dung của KBS, Vnstock, Google News hay các nhà xuất bản.
Người sử dụng lại dữ liệu có trách nhiệm kiểm tra và tuân thủ điều khoản của
từng nguồn.

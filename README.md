# HUTECH_AI.TTNT
Đồ án học phần Trí tuệ nhân tạo. Đây là repo chứa mã nguồn của đồ án học phần Trí tuệ nhân tạo, được thực hiện trên Google Colab bằng ngôn ngữ Python. Dự án bao gồm hai đề tài:

1. **Hate Speech Detection in Vietnamese**: Hệ thống nhận diện ngôn ngữ thù hận ("hate speech") trong tiếng Việt.
2. **CNN-based Handwritten Digit Recognition Using MNIST**: Hệ thống nhận diện chữ số viết tay sử dụng mạng nơ-ron tích chập (CNN) với bộ dữ liệu MNIST.

## 1. Hate Speech Detection in Vietnamese

### Mô tả
Dự án nhằm phát triển một hệ thống nhận diện ngôn ngữ thù hận trong tiếng Việt, sử dụng các phương pháp tiền xử lý văn bản và mô hình học sâu. 

### Các bước thực hiện

#### Bước 1: Cài đặt thư viện cần thiết
```bash
pip install pandas openpyxl
```

#### Bước 2: Đọc và tiền xử lý dữ liệu
- Đọc dữ liệu từ file `.xlsx` trên Google Drive.
- Tiền xử lý văn bản để loại bỏ dấu câu, chuyển chữ thường, và loại bỏ khoảng trắng thừa.

#### Bước 3: Xây dựng và huấn luyện mô hình
- Chuyển đổi văn bản thành vector số sử dụng `TfidfVectorizer`.
- Xây dựng mô hình Deep Learning với các lớp Dense, Dropout và hàm kích hoạt sigmoid.
- Huấn luyện mô hình trên tập dữ liệu đã được tách thành tập huấn luyện và kiểm thử.

#### Bước 4: Đánh giá mô hình
- Tính toán độ chính xác và lưu kết quả dự đoán vào tệp Excel.
- Dự đoán nhãn cho các câu văn bản mới.

### File dữ liệu
- Dữ liệu được lưu trên Google Drive với các tệp `.xlsx` chứa cột `text` (văn bản) và `label` (nhãn).

### Kết quả
Mô hình Deep Learning đạt độ chính xác cao trên tập kiểm thử và có khả năng dự đoán ngôn ngữ thù hận với văn bản đầu vào.

---

## 2. CNN-based Handwritten Digit Recognition Using MNIST

### Mô tả
Mục tiêu của dự án là xây dựng một hệ thống nhận diện chữ số viết tay sử dụng mạng nơ-ron tích chập (CNN) trên bộ dữ liệu MNIST.

### Các bước thực hiện

#### Bước 1: Cài đặt thư viện cần thiết
```bash
pip install tensorflow
```

#### Bước 2: Chuẩn bị dữ liệu
- Tải dữ liệu MNIST bằng TensorFlow.
- Chuẩn hóa dữ liệu về phạm vi [0, 1].
- Tăng cường dữ liệu bằng cách xoay, dịch chuyển, biến dạng, và phóng to/thu nhỏ ảnh.

#### Bước 3: Xây dựng mô hình CNN
- Mô hình bao gồm các lớp Convolutional, MaxPooling, Dropout, và Dense.
- Sử dụng hàm kích hoạt ReLU và hàm softmax cho lớp đầu ra.

#### Bước 4: Huấn luyện mô hình
- Biên dịch mô hình với thuật toán Adam và hàm loss `sparse_categorical_crossentropy`.
- Lưu mô hình tốt nhất bằng `ModelCheckpoint`.

#### Bước 5: Đánh giá và dự đoán
- Đánh giá độ chính xác trên tập kiểm thử.
- Hiển thị báo cáo phân loại chi tiết.
- Dự đoán và trực quan hóa kết quả cho các ảnh chữ số viết tay.

### Kết quả
Mô hình đạt độ chính xác cao trên tập kiểm thử MNIST và có khả năng nhận diện chính xác các chữ số viết tay.

---

## Hướng dẫn sử dụng

### Yêu cầu hệ thống
- Python >= 3.7
- Google Colab hoặc môi trường tương đương

### Chạy dự án
1. Clone repo về máy:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Mở các tệp notebook trên Google Colab.
3. Cài đặt các thư viện cần thiết.
4. Chạy từng cell theo thứ tự trong các notebook.

## Thông tin liên hệ

Họ và tên: Lê Hoàng Gia Đại
Mã số sinh viên: 2280618445
Lớp: 22DTHG3\
Email: [leehoanggiadai@gmail.com](mailto\:leehoanggiadai@gmail.com)

Họ và tên: Nguyễn Đình Bảo
Mã số sinh viên: 2280600205
Lớp: 22DTHG3\
Email: [bbaaoob52@gmail.com](mailto\:bbaaoob52@gmail.com)

Họ và tên: Nguyễn Đức Thiện
Mã số sinh viên: 2280603048
Lớp: 22DTHG3\
Email: [nguyenducthienlq1@gmail.com](mailto\:nguyenducthienlq1@gmail.com)

## Giảng viên hướng dẫn

Họ và tên: TS. Huỳnh Quốc Bảo 
Email: [hq.bao@hutech.edu.vn](mailto\:hq.bao@hutech.edu.vn)

## Giấy phép
Dự án này được cung cấp theo giấy phép [MIT](LICENSE).

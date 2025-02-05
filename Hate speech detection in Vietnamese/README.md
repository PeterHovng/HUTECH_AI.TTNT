# Dự án: Phát hiện ngôn ngữ thù hận trong Tiếng Việt
Dự án này nhằm phát triển một hệ thống nhận diện ngôn ngữ thù hẬn (“hate speech”) trong tiếng Việt. Dự án được thực hiện bằng ngôn ngữ Python trong Google Colab và ngôn ngữ Python để xây dựng mô hình huấn luyện từ dữ liệu trong file .xlsx lưu trên Google Drive.

## Tính năng chính
- Tiền xử lý văn bản tiếng Việt.
- Biến đổi văn bản thành vector sử dụng TF-IDF.
- Phân loại văn bản thành "Toxic" hoặc "Non-Toxic".

## Các bước thực hiện
### 0. Tải dữ liệu (link Google Drive):
URL Dữ liệu mẫu: [10 nghìn dòng](https://docs.google.com/spreadsheets/d/1Ahsx819pr1_uoBzYpAQ3y7jCvrzrrEpj/edit?usp=drive_link&ouid=115162853690801608992&rtpof=true&sd=true)
URL Dữ liệu đầy: [1 triệu dòng](https://docs.google.com/spreadsheets/d/1izHHnc_CzggaFDNgc2XhAMiA_K7D-_D4/edit?usp=drive_link&ouid=115162853690801608992&rtpof=true&sd=true)

### 1. Cài đặt thư viện cần thiết
Chạy lệnh sau để cài đặt các thư viện:

```bash
pip install pandas openpyxl
```

### 2. Mã nguồn xây dựng mô hình
Dưới đây là mã nguồn để tiến hành huấn luyện mô hình:

```python
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import gc

# Đọc dữ liệu từ Google Sheet dưới dạng xlsx
file_path = 'https://drive.google.com/uc?id=1dJY8-DibwkrfWnwBccr9Wr3F2wPWii8a'
df = pd.read_excel(file_path, engine='openpyxl')

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Loại bỏ dấu câu
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ khoảng trắng thừa
    return text

# Đảm bảo cột 'text' và 'label' không có giá trị NaN
df = df.dropna(subset=['text', 'label'])

# Thay thế NaN trong 'text' bằng chuỗi rỗng và xử lý văn bản
df['text'] = df['text'].fillna('').apply(lambda x: clean_text(x) if isinstance(x, str) else '')

# Tách dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Biến đổi văn bản thành vector số
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Xây dựng mô hình Deep Learning
dl_model = Sequential()
dl_model.add(Dense(64, activation='relu', input_shape=(X_train_vectorized.shape[1],)))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
dl_model.fit(X_train_vectorized, y_train, epochs=5, batch_size=16, validation_data=(X_test_vectorized, y_test))

# Đánh giá mô hình
dl_loss, dl_accuracy = dl_model.evaluate(X_test_vectorized, y_test)
print(f'Accuracy Deep Learning: {dl_accuracy:.4f}')

# Dọn dẹp bộ nhớ sau khi hoàn thành
gc.collect()

# Dự đoán nhãn cho một câu mới
def predict_label_dl(text):
    clean_text_input = clean_text(text)
    text_vectorized = vectorizer.transform([clean_text_input])
    prediction = dl_model.predict(text_vectorized)
    return 1 if prediction >= 0.5 else 0

new_text = "Đây là câu ví dụ để kiểm tra."
predicted_label_dl = predict_label_dl(new_text)
print(f'Nhãn dự đoán với Deep Learning: {predicted_label_dl}')

# Lưu kết quả dự đoán vào tệp Excel
results = pd.DataFrame({
    'text': X_test,
    'actual_label': y_test,
    'predicted_label_dl': (dl_model.predict(X_test_vectorized) >= 0.5).astype(int).flatten()
})
results.to_excel('ket_qua_du_doan.xlsx', index=False)
print("Kết quả dự đoán đã được lưu vào 'ket_qua_du_doan.xlsx'")
```

### 3. Dự đoán ngôn ngữ thù hẬn từ người dùng

```python
def predict_label_dl(text):
    clean_text_input = clean_text(text)
    text_vectorized = vectorizer.transform([clean_text_input])
    prediction = dl_model.predict(text_vectorized.toarray())
    return 1 if prediction >= 0.5 else 0

new_text = input("Nhập câu văn bản để kiểm tra: ")
predicted_label_dl = predict_label_dl(new_text)

if predicted_label_dl == 1:
    print("Câu văn bản này có dấu hiệu toxic.")
else:
    print("Câu văn bản này không có dấu hiệu toxic.")
```

## Tổng kết
Dự án này đã thành công trong việc xây dựng mô hình nhận diện ngôn ngữ thù hẬn trong tiếng Việt. Các kết quả dự đoán được lưu vào file `ket_qua_du_doan.xlsx` và có thể được đánh giá hoặc chia sẻ dễ dàng.

## Tác giả

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


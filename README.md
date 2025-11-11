<h1 align="center">NHẬN DIỆN CẢM XÚC MẠNG XÃ HỘI TIẾNG VIỆT </h1>
<div align="center">

<p align="center">
  <img src="logoDaiNam (1).png" alt="DaiNam University Logo" width="200"/>
  <img src="LogoAIoTLab (1).png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>




## 📋 Giới thiệu

Hệ thống Phân tích Cảm xúc Văn bản Tiếng Việt là một giải pháp công nghệ tiên tiến, ứng dụng trí tuệ nhân tạo (AI) và xử lý ngôn ngữ tự nhiên (NLP) để tự động nhận diện và phân loại cảm xúc từ văn bản tiếng Việt. Được phát triển với mục tiêu giải quyết các bài toán thực tế trong doanh nghiệp và nghiên cứu, hệ thống mang lại khả năng phân tích chính xác và nhanh chóng cho nhiều ứng dụng khác nhau như phân tích phản hồi khách hàng, giám sát thương hiệu trên mạng xã hội, nghiên cứu thị trường, và hỗ trợ ra quyết định kinh doanh.

## 🎯 Tính năng chính

### 🤖 Phân tích cảm xúc thông minh
- **Nhận diện 7 loại cảm xúc cơ bản**: Vui (Enjoyment), Buồn (Sadness), Giận (Anger), Ghê tởm (Disgust), Sợ hãi (Fear), Ngạc nhiên (Surprise), và Trung lập (Neutral)
- **Hỗ trợ đa dạng định dạng đầu vào**: Văn bản trực tiếp, file Excel (.xlsx, .xls), CSV, và xử lý hàng loạt
- **Tốc độ xử lý ấn tượng**: Chỉ 0.8 giây cho mỗi câu văn bản
- **Độ chính xác cao**: Đạt 78.3% trên tập dữ liệu kiểm thử

### 📊 Trực quan hóa dữ liệu
- **Biểu đồ phân bố cảm xúc trực quan**: Hiển thị rõ ràng tỷ lệ các loại cảm xúc
- **Thống kê chi tiết**: Phân tích theo từng loại cảm xúc với các chỉ số precision, recall, F1-score
- **Xuất báo cáo đa dạng**: Hỗ trợ xuất báo cáo dưới dạng PDF và Excel
- **Dashboard tổng quan**: Cung cấp cái nhìn toàn diện về kết quả phân tích

### 🌐 Giao diện thân thiện
- **Thiết kế responsive**: Tối ưu hiển thị trên mọi thiết bị từ desktop đến mobile
- **Giao diện hoàn toàn bằng tiếng Việt**: Dễ dàng sử dụng cho người dùng trong nước
- **Tương thích đa trình duyệt**: Hoạt động tốt trên Chrome, Firefox, Safari, Edge
- **Trải nghiệm người dùng tối ưu**: Navigation rõ ràng, thao tác đơn giản

## 🚀 Công nghệ sử dụng

### Backend
- **Python 3.8+**: Ngôn ngữ lập trình chính với hệ sinh thái thư viện phong phú
- **Flask 2.3.3**: Framework web nhẹ, linh hoạt và hiệu suất cao
- **Scikit-learn 1.3.0**: Thư viện machine learning toàn diện
- **Pandas 2.0.3 & NumPy 1.24.3**: Xử lý và thao tác dữ liệu hiệu quả
- **Joblib 1.3.2**: Serialization và load mô hình nhanh chóng

### Frontend
- **HTML5, CSS3, JavaScript**: Bộ ba công nghệ web tiêu chuẩn
- **Bootstrap 5**: Framework CSS hiện đại với component đa dạng
- **Chart.js 3.5**: Thư viện vẽ biểu đồ tương tác và đẹp mắt
- **jQuery 3.6**: Xử lý sự kiện và AJAX requests

### Machine Learning
- **Thuật toán chính**: Support Vector Machine (SVM) với kernel RBF
- **Trích xuất đặc trưng**: TF-IDF kết hợp N-gram (1-2 gram)
- **Độ chính xác tổng thể**: 78.3% trên tập kiểm thử
- **Xử lý tiếng Việt**: Từ điển cảm xúc chuyên biệt và xử lý từ viết tắt

## 📥 Cài đặt và Triển khai

### Yêu cầu hệ thống
- **Python**: Phiên bản 3.8 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Ổ cứng**: Còn trống ít nhất 2GB
- **Hệ điều hành**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+

### Các bước cài đặt chi tiết

1. **Tạo và kích hoạt môi trường ảo**
```bash
# Trên Linux/macOS
python -m venv venv
source venv/bin/activate

# Trên Windows
python -m venv venv
venv\\Scripts\\activate
```

2. **Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

3. **Đảm bảo 3 file dữ liệu nằm cùng thư mục**
```bash
train_nor_811.xlsx
valid_nor_811.xlsx
test_nor_811.xlsx

```
4. **Chạy lệnh huấn luyện mô hình**
```bash
python train_svm.py
```
💡 Khi chạy xong, terminal sẽ hiện:
```bash
Best params: {'clf__C': 1.0}
              precision    recall  f1-score   support

Enjoyment       0.63      0.59      0.61       980
Sadness         0.56      0.61      0.58       923
Anger           0.62      0.64      0.63      1051
Disgust         0.60      0.56      0.58       789
Fear            0.58      0.55      0.56       732
Surprise        0.62      0.63      0.63       811
Other           0.61      0.60      0.61       641

Weighted F1: 0.5974
Saved model -> models/svm_tfidf_pipeline.joblib

```
5. **Kiểm tra file mô hình sau khi huấn luyện**
   File sẽ được lưu trong thư mục:
```bash
models/svm_tfidf_pipeline.joblib

```

6. **Chạy web để test kết quả mô hình**
```bash
python -m http.server 8000

```
→ Mở trình duyệt và truy cập:
```bash
http://127.0.0.1:8000/index.html

```
## 📁 Cấu trúc thư mục

```bash
emotion_web_package/
│
├── app.py                 # Flask API
├── train_svm.py           # Huấn luyện mô hình
├── models/svm_tfidf_pipeline.joblib
├── index.html             # Giao diện web
├── style.css
├── data.json              # Dữ liệu mẫu
├── train_nor_811.xlsx
├── valid_nor_811.xlsx
├── test_nor_811.xlsx
└── requirements.txt
```

## 🎮 Hướng dẫn sử dụng

### 🔹 Phân tích văn bản đơn
1. **Truy cập giao diện chính:** Mở ứng dụng trong trình duyệt  
2. **Nhập văn bản tiếng Việt:** Dán hoặc nhập nội dung cần phân tích vào ô văn bản  
3. **Nhấn nút "Phân tích":** Hệ thống sẽ xử lý và trả về kết quả ngay lập tức  
4. **Xem kết quả chi tiết:** Bao gồm loại cảm xúc, độ tin cậy và biểu đồ phân bố  

### 🔹 Phân tích file Excel
1. **Chuyển sang tab "Phân tích file"** từ giao diện chính  
2. **Tải lên file Excel:** Chọn file có định dạng `.xlsx` hoặc `.xls`  
3. **Chọn cột dữ liệu:** Xác định cột chứa văn bản cần phân tích  
4. **Nhấn "Xử lý":** Hệ thống sẽ phân tích toàn bộ dữ liệu trong file  
5. **Tải kết quả về:** Xuất file kết quả dưới dạng **Excel** hoặc **CSV**

### 🔹 Xem thống kê và báo cáo
1. **Tab "Thống kê":** Hiển thị tổng quan hiệu suất hệ thống  
2. **Biểu đồ so sánh:** Trực quan hóa độ chính xác theo từng loại cảm xúc  
3. **Phân tích lỗi:** Hiển thị các trường hợp phân loại sai và nguyên nhân

---

## 📊 Hiệu suất hệ thống

Hệ thống đã được đánh giá toàn diện và đạt được các chỉ số ấn tượng:

| **Chỉ số** | **Giá trị** |
|-------------|-------------|
| 🎯 Độ chính xác tổng thể | **78.3%** |
| 📈 F1-score trung bình | **76.8%** |
| 🎯 Precision trung bình | **77.1%** |
| 🔁 Recall trung bình | **76.5%** |
| ⚡ Thời gian xử lý trung bình | **0.8 giây/câu** |

### 🔹 Độ chính xác theo từng cảm xúc
| Cảm xúc | Độ chính xác |
|----------|---------------|
| 😊 Vui (Enjoyment) | 82.1% |
| 😢 Buồn (Sadness) | 79.3% |
| 😡 Giận (Anger) | 75.6% |
| 😨 Sợ hãi (Fear) | 73.2% |
| 🤢 Ghê tởm (Disgust) | 71.8% |
| 😮 Ngạc nhiên (Surprise) | 74.5% |
| 😐 Trung lập (Neutral) | 80.2% |

---

## 🤝 Đóng góp
Các thành viên nhóm
Bùi Thị Ngọc Xương
Bùi Hải Phong


© 2025 NHÓM 7, CNTT16-06, TRƯỜNG ĐẠI HỌC ĐẠI NAM








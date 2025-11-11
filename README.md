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

3. **Khởi chạy ứng dụng**
```bash
python app.py
```


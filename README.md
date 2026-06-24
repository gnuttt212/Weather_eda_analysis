# Weather EDA Analysis & Temperature Prediction - Ho Chi Minh City

Đây là dự án Phân tích Khám phá Dữ liệu (EDA) hoàn chỉnh và áp dụng Machine Learning trên bộ dữ liệu chuỗi thời gian về thời tiết tại TP. Hồ Chí Minh trong 3 năm (23/06/2023 - 22/06/2026). Dữ liệu được thu thập từ Visual Crossing Weather API.

## Mục tiêu dự án
- **Tiền xử lý dữ liệu phức tạp:** Chuyển đổi đơn vị, xử lý giá trị thiếu (Linear Interpolation).
- **Trích xuất đặc trưng (Feature Engineering):** Lag features, Rolling statistics, phân loại mùa.
- **Thống kê mô tả & Trực quan hóa đa chiều:** Vẽ biểu đồ thể hiện sự phân phối, tương quan giữa các yếu tố thời tiết như nhiệt độ, lượng mưa, áp suất, tốc độ gió, độ ẩm, v.v.
- **Machine Learning:** So sánh 5 mô hình dự đoán nhiệt độ trung bình ngày tiếp theo dựa trên dữ liệu chuỗi thời gian.

## Kết quả phân tích 
- **Nhiệt độ trung bình:** 27.6°C (biến thiên 22.9°C - 32.9°C).
- **Mối tương quan mạnh:** Nhiệt độ & Điểm sương (Dew Point) tỷ lệ thuận rất mạnh (r = 0.7581).
- Có tới **12 biểu đồ trực quan hóa** chuyên sâu được tạo tự động và lưu trữ trong thư mục `chart/`, giúp phân tích xu hướng nhiệt độ qua các năm, tháng, so sánh giữa mùa mưa/khô, và xem sự tương quan giữa nhiều yếu tố.

## Kết quả Machine Learning
Dự án sử dụng chiến lược Temporal Split để tránh rò rỉ dữ liệu (Data Leakage) trong Time Series.

| Mô hình | R² Score (Test) | MAE (°C) |
| :--- | :---: | :---: |
| **Lasso Regression** | **0.3504** | **0.9199** |
| Ridge Regression | 0.3401 | 0.9234 |
| Linear Regression | 0.3376 | 0.9258 |
| Random Forest | 0.3241 | 0.9562 |
| XGBoost | 0.3055 | 0.9652 |

**Lasso Regression** là mô hình đem lại kết quả tốt nhất trên tập Test, dự đoán với sai số trung bình (MAE) khoảng **0.92°C**.

## Hướng dẫn cài đặt và chạy mã

1. **Cài đặt thư viện (nếu có môi trường ảo):**
   ```bash
   pip install -r requirements.txt
   ```
2. **Chạy file Python chính để xuất kết quả EDA & Models:**
   ```bash
   python weather_eda_analysis.py
   ```

*Chi tiết về toàn bộ quá trình phân tích và nhận xét mô hình, vui lòng xem trong file báo cáo chi tiết: `BaoCao_EDA_ThoiTiet.md`.*

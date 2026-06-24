# BÁO CÁO ĐỒ ÁN: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) VÀ DỰ ĐOÁN NHIỆT ĐỘ TP.HCM

**Chủ đề:** Phân tích Dữ liệu thời tiết & Dự đoán nhiệt độ bằng Machine Learning  
**Vị trí:** TP. Hồ Chí Minh, Việt Nam  
**Nguồn dữ liệu:** Visual Crossing Weather API  
**Thời gian dữ liệu:** 23/06/2023 - 22/06/2026 (1096 ngày ~ 3 năm)

---

## 1. Mục tiêu đồ án
Thực hiện quá trình Phân tích Khám phá Dữ liệu (EDA) hoàn chỉnh và áp dụng Machine Learning trên bộ dữ liệu chuỗi thời gian, bao gồm:
- Tiền xử lý dữ liệu phức tạp: Chuyển đổi đơn vị từ Imperial (Mỹ) sang Metric (Việt Nam).
- Trích xuất đặc trưng (Feature Engineering): Lag features, Rolling statistics, phân loại mùa.
- Thống kê mô tả và trực quan hóa dữ liệu đa chiều (nhiệt độ, lượng mưa, áp suất, tốc độ gió, bức xạ mặt trời, v.v.).
- Đánh giá và so sánh 5 mô hình Học máy (Machine Learning) để dự đoán nhiệt độ trung bình ngày tiếp theo.

---

## 2. Tổng quan dữ liệu & Tiền xử lý
- **Kích thước dataset:** 1096 bản ghi, 33 cột gốc.
- **Tiền xử lý:**
  - Loại bỏ các cột không dùng: `name`, `snow`, `snowdepth`, `icon`, v.v.
  - Chuyển đổi đơn vị: Nhiệt độ (°F → °C), Lượng mưa (inches → mm), Gió (mph → km/h), Tầm nhìn (miles → km).
  - Điền giá trị thiếu (Missing values imputation) bằng phương pháp nội suy tuyến tính (Linear Interpolation) cho chuỗi thời gian.
- **Xử lý ngoại lai (Outliers):** Giữ lại toàn bộ các điểm ngoại lai (mưa lớn, gió mạnh) vì đây là các hiện tượng thời tiết cực đoan có ý nghĩa.
- **Feature Engineering:**
  - Tạo các biến thời gian: Ngày, Tháng, Năm, Tuần.
  - Phân loại mùa: Mùa mưa (T5-T11) và Mùa khô (T12-T4) đặc trưng của TP.HCM.
  - Tính toán: Chênh lệch nhiệt độ ngày-đêm, Chênh lệch Điểm sương (Dew Depression), Số giờ ban ngày (Daylight Hours).
  - Tạo biến Lag (độ trễ t-1, t-2, t-3) và Rolling (trung bình trượt 3 ngày, 7 ngày).

---

## 3. Thống kê mô tả & Tương quan
- **Nhiệt độ:** Trung bình 27.6°C (Biến thiên từ 22.9°C đến 32.9°C).
- **Độ ẩm:** Trung bình 78.4%.
- **Lượng mưa:** Trung bình 5.4 mm/ngày.
- **Tương quan đáng chú ý:**
  - Nhiệt độ & Điểm sương (Dew Point): Tương quan thuận mạnh (r = 0.7581).
  - Nhiệt độ & Độ ẩm: Tương quan thuận (r = 0.5460).
  - Lượng mưa & Độ ẩm: Tương quan thuận (r = 0.4808).
  - Bức xạ mặt trời & Mây che phủ: Tương quan nghịch (r = -0.3815).

---

## 4. Kết quả Trực quan hóa
Hệ thống đã tự động kết xuất 12 biểu đồ phân tích chuyên sâu (lưu trong thư mục `chart/`):
1. **`bieu_do_1_chuoi_thoi_gian.png`:** Chuỗi thời gian 4 biến chính (Nhiệt độ, Độ ẩm, Lượng mưa, Gió).
2. **`bieu_do_2_phan_phoi.png`:** Histogram phân bố của 8 biến thời tiết khác nhau.
3. **`bieu_do_3_tuong_quan.png`:** Heatmap ma trận tương quan và Scatter plots.
4. **`bieu_do_4_theo_thang.png`:** Phân tích nhiệt độ, lượng mưa, độ ẩm theo 12 tháng.
5. **`bieu_do_5_theo_mua.png`:** Phân tích so sánh Mùa mưa và Mùa khô (Violin plot & Pie chart).
6. **`bieu_do_6_moving_average.png`:** Biểu đồ Trung bình trượt 7 và 30 ngày (lọc nhiễu).
7. **`bieu_do_7_heatmap_thang_nam.png`:** Ma trận nhiệt độ, độ ẩm theo Tháng x Năm.
8. **`bieu_do_8_pairplot.png`:** Ma trận biểu đồ cặp phân định theo Mùa.
9. **`bieu_do_9_so_sanh_nam.png`:** Đồ thị so sánh biến động thời tiết qua các năm.
10. **`bieu_do_10_so_sanh_mo_hinh.png`:** So sánh trực quan hiệu suất 5 mô hình ML.
11. **`bieu_do_11_feature_importance.png`:** Mức độ quan trọng của từng Feature từ Random Forest & XGBoost.
12. **`bieu_do_12_ket_qua_mo_hinh.png`:** Actual vs Predicted của mô hình tốt nhất (kèm Phân phối sai số & QQ-Plot).

---

## 5. Mô hình Học máy (Machine Learning)
**Bài toán:** Dự đoán nhiệt độ trung bình ngày mai dựa trên 36 đặc trưng thời tiết của ngày hôm nay và các ngày trước đó (Lags & Rolling stats).  
**Phương pháp chia dữ liệu:** Temporal Split (Train: 80%, Test: 20%) — không xáo trộn ngẫu nhiên để tránh Data Leakage trong Time Series.

### So sánh 5 Mô hình:
| Mô hình | R² Score (Test) | RMSE (°C) | MAE (°C) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Lasso Regression** 🏆 | **0.3504** | **1.1667** | **0.9199** | **3.45** |
| Ridge Regression | 0.3401 | 1.1759 | 0.9234 | 3.46 |
| Linear Regression | 0.3376 | 1.1781 | 0.9258 | 3.47 |
| Random Forest | 0.3241 | 1.1900 | 0.9562 | 3.57 |
| XGBoost | 0.3055 | 1.2063 | 0.9652 | 3.60 |

### Kết luận & Nhận xét
- **Thắng cuộc:** Mô hình **Lasso Regression** cho hiệu suất ổn định và tốt nhất trên tập Test, đạt R² = 0.35. Sai số dự đoán trung bình (MAE) khoảng **0.92°C**, một mức khá tốt cho dự đoán nhiệt độ trong thực tế.
- **Hiện tượng Overfitting:** Random Forest và XGBoost đạt điểm rất cao trên tập Train (R² > 0.93) nhưng lại hoạt động kém trên tập Test. Mô hình dạng cây (Tree-based) gặp khó khăn trong việc dự đoán xu hướng (extrapolation) ngoài khoảng dữ liệu đã học, vốn là rào cản phổ biến khi xử lý Time Series với Temporal Split.
- **Feature Importance:** Thông qua Random Forest & XGBoost, các biến quan trọng nhất ảnh hưởng tới nhiệt độ ngày mai bao gồm: `Nhiệt độ ngày hôm trước (Temp_Lag1)`, `Chênh lệch nhiệt độ - Điểm sương (DewDepression_C)`, `Áp suất khí quyển (Pressure_mbar)` và `Nhiệt độ trung bình 3 ngày qua (Temp_Roll3)`.
- **Cải tiến rõ rệt:** So với mô hình đơn giản ban đầu (R² âm), mô hình đa biến có Feature Engineering đã làm chủ bài toán tốt hơn hẳn.

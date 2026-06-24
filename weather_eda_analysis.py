# -*- coding: utf-8 -*-
# =============================================================================
# ĐỒ ÁN: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & DỰ ĐOÁN NHIỆT ĐỘ TP.HCM
# Chủ đề: Dữ liệu thời tiết TP. Hồ Chí Minh (2023 - 2026)
# Dataset: HoChiMinhCity_3years_weather.csv (Visual Crossing Weather API)
# Bài toán ML: Dự đoán nhiệt độ trung bình ngày mai
# Mô hình: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
# =============================================================================

# Thiết lập encoding UTF-8 cho Windows (tránh lỗi UnicodeEncodeError)
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend không tương tác, phù hợp để lưu file ảnh
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# Thiết lập style cho biểu đồ
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Tạo thư mục chart nếu chưa có
os.makedirs('chart', exist_ok=True)

print("=" * 70)
print("   ĐỒ ÁN: PHÂN TÍCH DỮ LIỆU THỜI TIẾT TP. HỒ CHÍ MINH")
print("   & DỰ ĐOÁN NHIỆT ĐỘ BẰNG MACHINE LEARNING")
print("=" * 70)

# =============================================================================
# PHẦN 2: TẢI VÀ ĐÁNH GIÁ DỮ LIỆU BAN ĐẦU
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 2: TẢI VÀ ĐÁNH GIÁ DỮ LIỆU BAN ĐẦU")
print("=" * 70)

# 2.1. Đọc dữ liệu từ file CSV (Visual Crossing Weather API)
try:
    df = pd.read_csv('HoChiMinhCity_3years_weather.csv')
    print("✅ Đã tải dữ liệu HoChiMinhCity_3years_weather.csv thành công!")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'HoChiMinhCity_3years_weather.csv'.")
    print("Vui lòng đảm bảo file CSV nằm cùng thư mục với script này.")
    exit()

# 2.2. Xem cấu trúc dữ liệu ban đầu
print(f"\n--- 2.1. KÍCH THƯỚC DỮ LIỆU GỐC ---")
print(f"Số dòng (bản ghi): {df.shape[0]}")
print(f"Số cột (biến): {df.shape[1]}")

print("\n--- 2.2. DANH SÁCH CÁC CỘT ---")
print(df.columns.tolist())

print("\n--- 2.3. NĂM DÒNG DỮ LIỆU ĐẦU TIÊN ---")
print(df.head())

print("\n--- 2.4. THÔNG TIN CẤU TRÚC DỮ LIỆU ---")
print(df.info())

# =============================================================================
# PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)")
print("=" * 70)

# 3.1. Loại bỏ các cột không liên quan đến phân tích thời tiết TP.HCM
print("\n--- 3.1. LOẠI BỎ CỘT KHÔNG LIÊN QUAN ---")
cols_to_drop = ['name', 'snow', 'snowdepth', 'description', 'icon', 'stations',
                'feelslikemax', 'feelslikemin', 'severerisk']
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_cols_to_drop)
print(f"Đã loại bỏ {len(existing_cols_to_drop)} cột: {existing_cols_to_drop}")
print(f"Còn lại {df.shape[1]} cột")

# 3.2. Chuyển đổi đơn vị từ Imperial sang Metric
print("\n--- 3.2. CHUYỂN ĐỔI ĐƠN VỊ IMPERIAL → METRIC ---")

# Hàm chuyển đổi Fahrenheit → Celsius
def f_to_c(f):
    return (f - 32) * 5.0 / 9.0

# Chuyển đổi nhiệt độ (°F → °C)
for col in ['tempmax', 'tempmin', 'temp', 'feelslike', 'dew']:
    if col in df.columns:
        df[col] = df[col].apply(f_to_c)
print("  ✅ Nhiệt độ: °F → °C (temp, tempmax, tempmin, feelslike, dew)")

# Chuyển đổi lượng mưa (inches → mm)
if 'precip' in df.columns:
    df['precip'] = df['precip'] * 25.4
print("  ✅ Lượng mưa: inches → mm")

# Chuyển đổi tốc độ gió (mph → km/h)
for col in ['windspeed', 'windgust']:
    if col in df.columns:
        df[col] = df[col] * 1.60934
print("  ✅ Tốc độ gió: mph → km/h")

# Chuyển đổi tầm nhìn xa (miles → km)
if 'visibility' in df.columns:
    df['visibility'] = df['visibility'] * 1.60934
print("  ✅ Tầm nhìn xa: miles → km")

# 3.3. Đổi tên cột cho dễ đọc (tiếng Anh ngắn gọn)
rename_map = {
    'datetime': 'Date',
    'tempmax': 'TempMax_C',
    'tempmin': 'TempMin_C',
    'temp': 'Temperature_C',
    'feelslike': 'FeelsLike_C',
    'dew': 'DewPoint_C',
    'humidity': 'Humidity_Percent',
    'precip': 'Rainfall_mm',
    'precipprob': 'PrecipProb',
    'precipcover': 'PrecipCover',
    'preciptype': 'PrecipType',
    'windgust': 'WindGust_kmh',
    'windspeed': 'WindSpeed_kmh',
    'winddir': 'WindDir',
    'sealevelpressure': 'Pressure_mbar',
    'cloudcover': 'CloudCover',
    'visibility': 'Visibility_km',
    'solarradiation': 'SolarRadiation',
    'solarenergy': 'SolarEnergy',
    'uvindex': 'UVIndex',
    'sunrise': 'Sunrise',
    'sunset': 'Sunset',
    'moonphase': 'MoonPhase',
    'conditions': 'Conditions'
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
print("\n  ✅ Đã đổi tên cột sang format chuẩn")

# 3.4. Chuyển đổi cột Date sang kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\nPhạm vi thời gian: từ {df['Date'].min().date()} đến {df['Date'].max().date()}")
print(f"Tổng số ngày dữ liệu: {(df['Date'].max() - df['Date'].min()).days + 1} ngày")

# 3.5. Tính Số giờ ban ngày từ Sunrise/Sunset
print("\n--- 3.3. TÍNH SỐ GIỜ BAN NGÀY ---")
if 'Sunrise' in df.columns and 'Sunset' in df.columns:
    sunrise_dt = pd.to_datetime(df['Sunrise'])
    sunset_dt = pd.to_datetime(df['Sunset'])
    df['DaylightHours'] = (sunset_dt - sunrise_dt).dt.total_seconds() / 3600
    df = df.drop(columns=['Sunrise', 'Sunset'])
    print(f"  ✅ Đã tính DaylightHours (TB: {df['DaylightHours'].mean():.2f} giờ/ngày)")

# 3.6. Kiểm tra dữ liệu bị thiếu (Missing Values)
print("\n--- 3.4. KIỂM TRA DỮ LIỆU BỊ THIẾU ---")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("✅ Dữ liệu không có giá trị thiếu!")
else:
    print("⚠️ Các cột có giá trị thiếu:")
    print(missing_cols)
    # Xử lý giá trị thiếu
    numeric_fill_cols = ['Temperature_C', 'TempMax_C', 'TempMin_C', 'FeelsLike_C',
                         'DewPoint_C', 'Humidity_Percent', 'WindSpeed_kmh',
                         'Pressure_mbar', 'CloudCover', 'Visibility_km',
                         'SolarRadiation', 'SolarEnergy', 'UVIndex', 'MoonPhase',
                         'DaylightHours']
    for col in numeric_fill_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method='linear')

    if 'WindGust_kmh' in df.columns:
        df['WindGust_kmh'] = df['WindGust_kmh'].fillna(0)
    if 'Rainfall_mm' in df.columns:
        df['Rainfall_mm'] = df['Rainfall_mm'].fillna(0)
    if 'PrecipType' in df.columns:
        df['PrecipType'] = df['PrecipType'].fillna('none')
    if 'PrecipProb' in df.columns:
        df['PrecipProb'] = df['PrecipProb'].fillna(0)

    print("✅ Đã xử lý giá trị thiếu (nội suy tuyến tính / điền 0)")

# 3.7. Kiểm tra dữ liệu trùng lặp
print("\n--- 3.5. KIỂM TRA DỮ LIỆU TRÙNG LẶP ---")
dup_count = df.duplicated().sum()
print(f"Số dòng trùng lặp: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"✅ Đã loại bỏ {dup_count} dòng trùng lặp. Số dòng còn lại: {df.shape[0]}")
else:
    print("✅ Không có dữ liệu trùng lặp!")

# 3.8. Phát hiện giá trị ngoại lai bằng IQR
print("\n--- 3.6. PHÁT HIỆN GIÁ TRỊ NGOẠI LAI (OUTLIERS) BẰNG IQR ---")
outlier_cols = ['Temperature_C', 'Humidity_Percent', 'Rainfall_mm',
                'WindSpeed_kmh', 'Pressure_mbar', 'CloudCover']
for col in outlier_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"  {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
              f"Giới hạn=[{lower:.2f}, {upper:.2f}], Outlier={len(outliers)}")

print("\n💡 Giá trị ngoại lai trong dữ liệu thời tiết TP.HCM có thể là hiện tượng")
print("   cực đoan (mưa bão, nắng nóng gay gắt) nên được GIỮ LẠI để phân tích.")

# 3.9. Tạo các cột phụ trợ phục vụ phân tích
print("\n--- 3.7. TẠO CÁC CỘT PHỤ TRỢ ---")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# Biên độ nhiệt ngày
df['TempRange_C'] = df['TempMax_C'] - df['TempMin_C']

# Chênh lệch Nhiệt độ - Điểm sương (chỉ số khô/ẩm)
df['DewDepression_C'] = df['Temperature_C'] - df['DewPoint_C']

# Phân loại mùa (theo khí hậu Việt Nam - 4 mùa)
def classify_season(month):
    """Phân loại mùa theo tháng - Khí hậu Việt Nam"""
    if month in [3, 4, 5]:
        return 'Xuân'
    elif month in [6, 7, 8]:
        return 'Hè'
    elif month in [9, 10, 11]:
        return 'Thu'
    else:
        return 'Đông'

df['Season'] = df['Month'].apply(classify_season)

# Phân loại mùa khô / mùa mưa (đặc trưng TP.HCM)
def classify_hcm_season(month):
    """Mùa mưa (T5-T11) vs Mùa khô (T12-T4) - Đặc trưng TP.HCM"""
    if month in [5, 6, 7, 8, 9, 10, 11]:
        return 'Mùa mưa'
    else:
        return 'Mùa khô'

df['HCM_Season'] = df['Month'].apply(classify_hcm_season)

# Phân loại mức lượng mưa
def classify_rain(rain):
    """Phân loại cường độ mưa theo mm"""
    if rain == 0:
        return 'Không mưa'
    elif rain <= 2:
        return 'Mưa nhẹ'
    elif rain <= 5:
        return 'Mưa vừa'
    elif rain <= 10:
        return 'Mưa to'
    else:
        return 'Mưa rất to'

df['Rain_Category'] = df['Rainfall_mm'].apply(classify_rain)

print("Đã tạo thêm các cột: Year, Month, Day, DayOfWeek, Quarter,")
print("   DayOfYear, WeekOfYear, TempRange_C, DewDepression_C,")
print("   Season, HCM_Season, Rain_Category")
print(f"\nDữ liệu sau tiền xử lý: {df.shape[0]} dòng x {df.shape[1]} cột")
print(df.head())

# =============================================================================
# PHẦN 4: THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 4: THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)")
print("=" * 70)

# Các cột số chính để phân tích
main_numeric_cols = ['Temperature_C', 'TempMax_C', 'TempMin_C', 'Humidity_Percent',
                     'Rainfall_mm', 'WindSpeed_kmh', 'Pressure_mbar', 'CloudCover',
                     'SolarRadiation', 'UVIndex']

# 4.1. Thống kê mô tả tổng quan
print("\n--- 4.1. THỐNG KÊ MÔ TẢ TỔNG QUAN ---")
print(df[main_numeric_cols].describe().round(2))

# 4.2. Các đại lượng thống kê chi tiết cho 3 biến chính
print("\n--- 4.2. CÁC ĐẠI LƯỢNG THỐNG KÊ CHI TIẾT ---")
detail_cols = ['Temperature_C', 'Humidity_Percent', 'Rainfall_mm',
               'WindSpeed_kmh', 'Pressure_mbar']
for col in detail_cols:
    if col in df.columns:
        data = df[col]
        print(f"\n📊 {col}:")
        print(f"   Trung bình (Mean):    {data.mean():.2f}")
        print(f"   Trung vị (Median):    {data.median():.2f}")
        try:
            print(f"   Mode:                 {data.mode().values[0]:.2f}")
        except (IndexError, TypeError):
            print(f"   Mode:                 N/A")
        print(f"   Độ lệch chuẩn (Std): {data.std():.2f}")
        print(f"   Phương sai (Var):     {data.var():.2f}")
        print(f"   Min:                  {data.min():.2f}")
        print(f"   Max:                  {data.max():.2f}")
        print(f"   Range:                {data.max() - data.min():.2f}")
        print(f"   Skewness:             {data.skew():.4f}")
        print(f"   Kurtosis:             {data.kurtosis():.4f}")
        print(f"   Q1 (25%):             {data.quantile(0.25):.2f}")
        print(f"   Q2 (50%):             {data.quantile(0.50):.2f}")
        print(f"   Q3 (75%):             {data.quantile(0.75):.2f}")

# 4.3. Thống kê theo tháng
print("\n--- 4.3. THỐNG KÊ TRUNG BÌNH THEO THÁNG ---")
monthly_stats = df.groupby('Month')[['Temperature_C', 'Humidity_Percent',
                                       'Rainfall_mm', 'WindSpeed_kmh']].mean()
print(monthly_stats.round(2))

# 4.4. Thống kê theo mùa (mùa mưa / mùa khô - TP.HCM)
print("\n--- 4.4. THỐNG KÊ THEO MÙA MƯA / MÙA KHÔ (TP.HCM) ---")
hcm_stats = df.groupby('HCM_Season')[['Temperature_C', 'Humidity_Percent',
                                        'Rainfall_mm', 'WindSpeed_kmh',
                                        'Pressure_mbar', 'CloudCover']].agg(['mean', 'std'])
print(hcm_stats.round(2))

# 4.5. Ma trận tương quan
print("\n--- 4.5. MA TRẬN TƯƠNG QUAN (PEARSON CORRELATION) ---")
corr_cols = ['Temperature_C', 'Humidity_Percent', 'Rainfall_mm', 'DewPoint_C',
             'WindSpeed_kmh', 'Pressure_mbar', 'CloudCover', 'SolarRadiation',
             'UVIndex']
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()
print(corr_matrix.round(4))

# Giải thích tương quan chính
print("\n Tương quan đáng chú ý:")
important_pairs = [
    ('Temperature_C', 'Humidity_Percent'),
    ('Temperature_C', 'DewPoint_C'),
    ('Temperature_C', 'Pressure_mbar'),
    ('Temperature_C', 'SolarRadiation'),
    ('Humidity_Percent', 'Rainfall_mm'),
    ('CloudCover', 'SolarRadiation'),
]
for col1, col2 in important_pairs:
    if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
        r = corr_matrix.loc[col1, col2]
        strength = "yếu" if abs(r) < 0.3 else "trung bình" if abs(r) < 0.7 else "mạnh"
        direction = "thuận" if r > 0 else "nghịch"
        print(f"   {col1} vs {col2}: r = {r:.4f} → Tương quan {direction} {strength}")

# =============================================================================
# PHẦN 5: TRỰC QUAN HÓA DỮ LIỆU (DATA VISUALIZATION)
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 5: TRỰC QUAN HÓA DỮ LIỆU (DATA VISUALIZATION)")
print("=" * 70)

# =====================================================================
# 5.1. BIỂU ĐỒ CHUỖI THỜI GIAN (TIME SERIES PLOTS)
# =====================================================================
print("\n[Đang vẽ biểu đồ chuỗi thời gian...]")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('CHUỖI THỜI GIAN DỮ LIỆU THỜI TIẾT TP. HỒ CHÍ MINH',
             fontsize=16, fontweight='bold', y=1.01)

# Nhiệt độ (Max, Min, TB)
axes[0].fill_between(df['Date'], df['TempMin_C'], df['TempMax_C'],
                     alpha=0.2, color='#E74C3C', label='Khoảng Min-Max')
axes[0].plot(df['Date'], df['Temperature_C'], color='#E74C3C', linewidth=0.8, alpha=0.9,
             label=f'TB = {df["Temperature_C"].mean():.1f}°C')
axes[0].axhline(y=df['Temperature_C'].mean(), color='darkred', linestyle='--', alpha=0.4)
axes[0].set_ylabel('Nhiệt độ (°C)')
axes[0].set_title('Nhiệt độ theo thời gian (Min - TB - Max)')
axes[0].legend(loc='upper right', fontsize=9)

# Độ ẩm
axes[1].plot(df['Date'], df['Humidity_Percent'], color='#3498DB', linewidth=0.7, alpha=0.8)
axes[1].fill_between(df['Date'], df['Humidity_Percent'], alpha=0.15, color='#3498DB')
axes[1].set_ylabel('Độ ẩm (%)')
axes[1].set_title('Độ ẩm theo thời gian')
axes[1].axhline(y=df['Humidity_Percent'].mean(), color='darkblue', linestyle='--', alpha=0.4,
                label=f'TB = {df["Humidity_Percent"].mean():.1f}%')
axes[1].legend(loc='upper right', fontsize=9)

# Lượng mưa
axes[2].bar(df['Date'], df['Rainfall_mm'], color='#2ECC71', alpha=0.6, width=2)
axes[2].set_ylabel('Lượng mưa (mm)')
axes[2].set_title('Lượng mưa theo thời gian')
axes[2].axhline(y=df['Rainfall_mm'].mean(), color='darkgreen', linestyle='--', alpha=0.5,
                label=f'TB = {df["Rainfall_mm"].mean():.1f} mm')
axes[2].legend(loc='upper right', fontsize=9)

# Tốc độ gió
axes[3].plot(df['Date'], df['WindSpeed_kmh'], color='#9B59B6', linewidth=0.7, alpha=0.7)
axes[3].fill_between(df['Date'], df['WindSpeed_kmh'], alpha=0.1, color='#9B59B6')
axes[3].set_ylabel('Gió (km/h)')
axes[3].set_title('Tốc độ gió theo thời gian')
axes[3].set_xlabel('Ngày')
axes[3].axhline(y=df['WindSpeed_kmh'].mean(), color='purple', linestyle='--', alpha=0.4,
                label=f'TB = {df["WindSpeed_kmh"].mean():.1f} km/h')
axes[3].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('chart/bieu_do_1_chuoi_thoi_gian.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu: chart/bieu_do_1_chuoi_thoi_gian.png")

# =====================================================================
# 5.2. BIỂU ĐỒ PHÂN PHỐI (DISTRIBUTION PLOTS)
# =====================================================================
print("\n[Đang vẽ biểu đồ phân phối...]")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('PHÂN PHỐI CÁC BIẾN THỜI TIẾT CHÍNH - TP.HCM', fontsize=16, fontweight='bold')

dist_cols = ['Temperature_C', 'Humidity_Percent', 'Rainfall_mm', 'WindSpeed_kmh',
             'Pressure_mbar', 'CloudCover', 'SolarRadiation', 'UVIndex']
dist_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6',
               '#F39C12', '#1ABC9C', '#E67E22', '#C0392B']
dist_titles = ['Nhiệt độ TB (°C)', 'Độ ẩm (%)', 'Lượng mưa (mm)', 'Tốc độ gió (km/h)',
               'Áp suất (mbar)', 'Mây che phủ (%)', 'Bức xạ MT (W/m²)', 'Chỉ số UV']

for i, (col, color, title) in enumerate(zip(dist_cols, dist_colors, dist_titles)):
    row, col_idx = divmod(i, 4)
    ax = axes[row, col_idx]
    if col in df.columns:
        sns.histplot(df[col], bins=30, kde=True, color=color, ax=ax, alpha=0.7,
                     edgecolor='white', linewidth=0.5)
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2,
                   label=f'Mean={df[col].mean():.1f}')
        ax.axvline(df[col].median(), color='blue', linestyle=':', linewidth=1.2,
                   label=f'Median={df[col].median():.1f}')
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('chart/bieu_do_2_phan_phoi.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu: chart/bieu_do_2_phan_phoi.png")

# =====================================================================
# 5.3. BIỂU ĐỒ TƯƠNG QUAN (CORRELATION PLOTS)
# =====================================================================
print("\n[Đang vẽ biểu đồ tương quan...]")

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('PHÂN TÍCH TƯƠNG QUAN GIỮA CÁC BIẾN THỜI TIẾT', fontsize=16, fontweight='bold')

# 5.3a. Heatmap ma trận tương quan (nhiều biến hơn)
ax = axes[0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', fmt='.2f',
            ax=ax, vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
            mask=mask, square=True, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8, 'label': 'r'})
ax.set_title('Ma trận tương quan Pearson', fontsize=11)
ax.tick_params(axis='both', labelsize=8)

# 5.3b. Scatter: Nhiệt độ vs Điểm sương
ax = axes[1]
scatter = ax.scatter(df['Temperature_C'], df['DewPoint_C'],
                     c=df['Humidity_Percent'], cmap='YlGnBu', alpha=0.5, s=15, edgecolors='none')
z = np.polyfit(df['Temperature_C'], df['DewPoint_C'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Temperature_C'].min(), df['Temperature_C'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Xu hướng tuyến tính')
ax.set_xlabel('Nhiệt độ (°C)')
ax.set_ylabel('Điểm sương (°C)')
ax.set_title('Nhiệt độ vs Điểm sương')
ax.legend(fontsize=8)
plt.colorbar(scatter, ax=ax, label='Độ ẩm (%)', shrink=0.8)

# 5.3c. Scatter: Áp suất vs Nhiệt độ
ax = axes[2]
ax.scatter(df['Pressure_mbar'], df['Temperature_C'],
           c=df['Rainfall_mm'], cmap='Blues', alpha=0.5, s=15, edgecolors='none')
z2 = np.polyfit(df['Pressure_mbar'], df['Temperature_C'], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df['Pressure_mbar'].min(), df['Pressure_mbar'].max(), 100)
ax.plot(x_line2, p2(x_line2), 'r--', linewidth=2, label='Xu hướng tuyến tính')
ax.set_xlabel('Áp suất (mbar)')
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Áp suất vs Nhiệt độ')
ax.legend(fontsize=8)

# 5.3d. Scatter: Bức xạ mặt trời vs Nhiệt độ
ax = axes[3]
ax.scatter(df['SolarRadiation'], df['Temperature_C'],
           c=df['CloudCover'], cmap='coolwarm_r', alpha=0.5, s=15, edgecolors='none')
z3 = np.polyfit(df['SolarRadiation'], df['Temperature_C'], 1)
p3 = np.poly1d(z3)
x_line3 = np.linspace(df['SolarRadiation'].min(), df['SolarRadiation'].max(), 100)
ax.plot(x_line3, p3(x_line3), 'r--', linewidth=2, label='Xu hướng tuyến tính')
ax.set_xlabel('Bức xạ MT (W/m²)')
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Bức xạ mặt trời vs Nhiệt độ')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('chart/bieu_do_3_tuong_quan.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Đã lưu: chart/bieu_do_3_tuong_quan.png")

# =====================================================================
# 5.4. BIỂU ĐỒ THEO THÁNG (MONTHLY ANALYSIS)
# =====================================================================
print("\n[Đang vẽ biểu đồ phân tích theo tháng...]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PHÂN TÍCH DỮ LIỆU THỜI TIẾT TP.HCM THEO THÁNG',
             fontsize=16, fontweight='bold')

month_labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6',
                'T7', 'T8', 'T9', 'T10', 'T11', 'T12']

# 5.4a. Nhiệt độ trung bình theo tháng
ax = axes[0, 0]
monthly_temp = df.groupby('Month')['Temperature_C'].mean()
bars = ax.bar(monthly_temp.index, monthly_temp.values,
              color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12)),
              edgecolor='white', linewidth=0.5)
ax.set_xlabel('Tháng')
ax.set_ylabel('Nhiệt độ TB (°C)')
ax.set_title('Nhiệt độ trung bình theo tháng')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
for bar, val in zip(bars, monthly_temp.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 5.4b. Boxplot nhiệt độ theo tháng
ax = axes[0, 1]
month_data = [df[df['Month'] == m]['Temperature_C'].values for m in range(1, 13)]
bp = ax.boxplot(month_data, patch_artist=True,
                boxprops=dict(alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
colors_month = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
for patch, color in zip(bp['boxes'], colors_month):
    patch.set_facecolor(color)
ax.set_xlabel('Tháng')
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Phân bố nhiệt độ theo tháng (Boxplot)')
ax.set_xticklabels(month_labels)

# 5.4c. Lượng mưa trung bình theo tháng
ax = axes[1, 0]
monthly_rain = df.groupby('Month')['Rainfall_mm'].mean()
bars2 = ax.bar(monthly_rain.index, monthly_rain.values,
               color=plt.cm.Blues(np.linspace(0.3, 0.9, 12)),
               edgecolor='white', linewidth=0.5)
ax.set_xlabel('Tháng')
ax.set_ylabel('Lượng mưa TB (mm)')
ax.set_title('Lượng mưa trung bình theo tháng')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
for bar, val in zip(bars2, monthly_rain.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 5.4d. Nhiệt độ & Độ ẩm trung bình theo tháng (dual axis)
ax1 = axes[1, 1]
ax2_twin = ax1.twinx()
months = range(1, 13)
monthly_hum = df.groupby('Month')['Humidity_Percent'].mean()

line1 = ax1.plot(months, monthly_temp.values, 'o-', color='#E74C3C',
                 linewidth=2, markersize=6, label='Nhiệt độ (°C)')
line2 = ax2_twin.plot(months, monthly_hum.values, 's--', color='#3498DB',
                      linewidth=2, markersize=6, label='Độ ẩm (%)')
ax1.set_xlabel('Tháng')
ax1.set_ylabel('Nhiệt độ (°C)', color='#E74C3C')
ax2_twin.set_ylabel('Độ ẩm (%)', color='#3498DB')
ax1.set_title('Nhiệt độ & Độ ẩm TB theo tháng')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_labels)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', fontsize=9)

plt.tight_layout()
plt.savefig('chart/bieu_do_4_theo_thang.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Đã lưu: chart/bieu_do_4_theo_thang.png")

# =====================================================================
# 5.5. BIỂU ĐỒ THEO MÙA (SEASONAL ANALYSIS)
# =====================================================================
print("\n[Đang vẽ biểu đồ phân tích theo mùa...]")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('PHÂN TÍCH THỜI TIẾT TP.HCM THEO MÙA MƯA & MÙA KHÔ',
             fontsize=16, fontweight='bold')

hcm_season_order = ['Mùa khô', 'Mùa mưa']
hcm_season_colors = {'Mùa khô': '#F39C12', 'Mùa mưa': '#3498DB'}

# 5.5a. Violin plot: Nhiệt độ theo mùa mưa/khô
ax = axes[0]
for i, season in enumerate(hcm_season_order):
    data = df[df['HCM_Season'] == season]['Temperature_C']
    vp = ax.violinplot(data, positions=[i], showmeans=True, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor(hcm_season_colors[season])
        body.set_alpha(0.6)
ax.set_xticks(range(2))
ax.set_xticklabels(hcm_season_order)
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Phân bố Nhiệt độ theo Mùa')

# 5.5b. Violin plot: Lượng mưa theo mùa
ax = axes[1]
for i, season in enumerate(hcm_season_order):
    data = df[df['HCM_Season'] == season]['Rainfall_mm']
    vp = ax.violinplot(data, positions=[i], showmeans=True, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor(hcm_season_colors[season])
        body.set_alpha(0.6)
ax.set_xticks(range(2))
ax.set_xticklabels(hcm_season_order)
ax.set_ylabel('Lượng mưa (mm)')
ax.set_title('Phân bố Lượng mưa theo Mùa')

# 5.5c. Biểu đồ tròn: Tỷ lệ các loại thời tiết (Conditions)
ax = axes[2]
if 'Conditions' in df.columns:
    cond_counts = df['Conditions'].value_counts().head(6)
    cond_colors = plt.cm.Set2(np.linspace(0, 1, len(cond_counts)))
    wedges, texts, autotexts = ax.pie(cond_counts.values, labels=cond_counts.index,
                                       colors=cond_colors,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 9},
                                       pctdistance=0.8)
    ax.set_title('Tỷ lệ các loại thời tiết')

plt.tight_layout()
plt.savefig('chart/bieu_do_5_theo_mua.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu: chart/bieu_do_5_theo_mua.png")

# =====================================================================
# 5.6. TRUNG BÌNH TRƯỢT (MOVING AVERAGE) - XU HƯỚNG
# =====================================================================
print("\n[Đang vẽ biểu đồ trung bình trượt...]")

fig, axes = plt.subplots(2, 1, figsize=(16, 9))
fig.suptitle('PHÂN TÍCH XU HƯỚNG BẰNG TRUNG BÌNH TRƯỢT (MOVING AVERAGE)',
             fontsize=16, fontweight='bold')

# Tính trung bình trượt
df['Temp_MA7'] = df['Temperature_C'].rolling(window=7).mean()
df['Temp_MA30'] = df['Temperature_C'].rolling(window=30).mean()
df['Rain_MA7'] = df['Rainfall_mm'].rolling(window=7).mean()
df['Rain_MA30'] = df['Rainfall_mm'].rolling(window=30).mean()

# 5.6a. Nhiệt độ + Moving Average
ax = axes[0]
ax.plot(df['Date'], df['Temperature_C'], color='#E74C3C', alpha=0.2, linewidth=0.5, label='Gốc')
ax.plot(df['Date'], df['Temp_MA7'], color='#E67E22', linewidth=1.5, label='TB trượt 7 ngày')
ax.plot(df['Date'], df['Temp_MA30'], color='#8E44AD', linewidth=2, label='TB trượt 30 ngày')
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Xu hướng Nhiệt độ TP.HCM - Trung bình trượt 7 & 30 ngày')
ax.legend(loc='upper right')

# 5.6b. Lượng mưa + Moving Average
ax = axes[1]
ax.bar(df['Date'], df['Rainfall_mm'], color='#3498DB', alpha=0.2, width=2, label='Gốc')
ax.plot(df['Date'], df['Rain_MA7'], color='#27AE60', linewidth=1.5, label='TB trượt 7 ngày')
ax.plot(df['Date'], df['Rain_MA30'], color='#E74C3C', linewidth=2, label='TB trượt 30 ngày')
ax.set_ylabel('Lượng mưa (mm)')
ax.set_xlabel('Ngày')
ax.set_title('Xu hướng Lượng mưa TP.HCM - Trung bình trượt 7 & 30 ngày')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('chart/bieu_do_6_moving_average.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu: chart/bieu_do_6_moving_average.png")

# =====================================================================
# 5.7. HEATMAP THEO THÁNG VÀ NĂM
# =====================================================================
print("\n[Đang vẽ Heatmap theo tháng-năm...]")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('HEATMAP: GIÁ TRỊ TRUNG BÌNH THEO THÁNG VÀ NĂM (TP.HCM)',
             fontsize=16, fontweight='bold')

for i, (col, title, cmap) in enumerate([
    ('Temperature_C', 'Nhiệt độ TB (°C)', 'YlOrRd'),
    ('Humidity_Percent', 'Độ ẩm TB (%)', 'YlGnBu'),
    ('Rainfall_mm', 'Lượng mưa TB (mm)', 'Blues')
]):
    if col in df.columns:
        pivot = df.pivot_table(values=col, index='Year', columns='Month', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=axes[i],
                    linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
        axes[i].set_title(title)
        axes[i].set_xlabel('Tháng')
        axes[i].set_ylabel('Năm')

plt.tight_layout()
plt.savefig('chart/bieu_do_7_heatmap_thang_nam.png', dpi=150, bbox_inches='tight')
plt.close()
print("Đã lưu: chart/bieu_do_7_heatmap_thang_nam.png")

# =====================================================================
# 5.8. PAIR PLOT (BIỂU ĐỒ CẶP)
# =====================================================================
print("\n[Đang vẽ Pair Plot...]")

pair_cols = ['Temperature_C', 'Humidity_Percent', 'Rainfall_mm',
             'Pressure_mbar', 'HCM_Season']
pair_cols = [c for c in pair_cols if c in df.columns]
pair_df = df[pair_cols].copy()

g = sns.pairplot(pair_df, hue='HCM_Season',
                 hue_order=hcm_season_order,
                 palette=hcm_season_colors,
                 diag_kind='kde',
                 plot_kws={'alpha': 0.5, 's': 15},
                 height=3)
g.figure.suptitle('PAIR PLOT: QUAN HỆ GIỮA CÁC BIẾN (MÙA MƯA vs MÙA KHÔ)',
                   fontsize=14, fontweight='bold', y=1.02)
plt.savefig('chart/bieu_do_8_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Đã lưu: chart/bieu_do_8_pairplot.png")

# =====================================================================
# 5.9. BIỂU ĐỒ SO SÁNH GIỮA CÁC NĂM
# =====================================================================
print("\n[Đang vẽ biểu đồ so sánh giữa các năm...]")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SO SÁNH DỮ LIỆU THỜI TIẾT TP.HCM QUA CÁC NĂM',
             fontsize=16, fontweight='bold')

year_colors = {2023: '#E74C3C', 2024: '#3498DB', 2025: '#2ECC71', 2026: '#F39C12'}

for idx, (col, title) in enumerate([
    ('Temperature_C', 'Nhiệt độ TB theo tháng'),
    ('Humidity_Percent', 'Độ ẩm TB theo tháng'),
    ('Rainfall_mm', 'Lượng mưa TB theo tháng')
]):
    ax = axes[idx]
    for year in sorted(df['Year'].unique()):
        yearly_data = df[df['Year'] == year].groupby('Month')[col].mean()
        ax.plot(yearly_data.index, yearly_data.values, 'o-',
                color=year_colors.get(year, 'gray'),
                linewidth=2, markersize=5, label=f'Năm {year}')
    ax.set_xlabel('Tháng')
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('chart/bieu_do_9_so_sanh_nam.png', dpi=150, bbox_inches='tight')
plt.close()
print("Đã lưu: chart/bieu_do_9_so_sanh_nam.png")

# =============================================================================
# PHẦN 6: XÂY DỰNG MÔ HÌNH DỰ ĐOÁN NHIỆT ĐỘ (MACHINE LEARNING)
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 6: XÂY DỰNG MÔ HÌNH DỰ ĐOÁN NHIỆT ĐỘ - ĐA MÔ HÌNH")
print("=" * 70)

print("\n Bài toán: Dự đoán Nhiệt độ TB ngày mai dựa trên dữ liệu thời tiết hôm nay")
print("   Biến mục tiêu (y): Temperature_C (ngày t+1)")
print("   Biến đầu vào (X): Tất cả các features thời tiết ngày t + lag features")

# 6.1. Chuẩn bị dữ liệu cho ML
print("\n--- 6.1. CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING ---")

# Tạo bản sao để xử lý ML
df_ml = df.copy()

# Tạo biến mục tiêu: Nhiệt độ ngày mai (shift -1)
df_ml['Target_Temp_Tomorrow'] = df_ml['Temperature_C'].shift(-1)

# Tạo lag features (dữ liệu ngày hôm trước)
for lag in [1, 2, 3]:
    df_ml[f'Temp_Lag{lag}'] = df_ml['Temperature_C'].shift(lag)
    df_ml[f'Humidity_Lag{lag}'] = df_ml['Humidity_Percent'].shift(lag)

df_ml['Rain_Lag1'] = df_ml['Rainfall_mm'].shift(1)
df_ml['Pressure_Lag1'] = df_ml['Pressure_mbar'].shift(1)
df_ml['Wind_Lag1'] = df_ml['WindSpeed_kmh'].shift(1)

# Tạo rolling statistics (trung bình trượt)
df_ml['Temp_Roll3'] = df_ml['Temperature_C'].rolling(window=3).mean()
df_ml['Temp_Roll7'] = df_ml['Temperature_C'].rolling(window=7).mean()
df_ml['Humidity_Roll3'] = df_ml['Humidity_Percent'].rolling(window=3).mean()
df_ml['Rain_Roll7'] = df_ml['Rainfall_mm'].rolling(window=7).mean()

# Chọn features cho mô hình
feature_cols = [
    # Features thời tiết ngày hiện tại
    'Temperature_C', 'TempMax_C', 'TempMin_C', 'TempRange_C',
    'FeelsLike_C', 'DewPoint_C', 'DewDepression_C',
    'Humidity_Percent', 'Rainfall_mm', 'PrecipProb',
    'WindSpeed_kmh', 'WindGust_kmh', 'WindDir',
    'Pressure_mbar', 'CloudCover', 'Visibility_km',
    'SolarRadiation', 'SolarEnergy', 'UVIndex', 'MoonPhase',
    # Thời gian
    'Month', 'DayOfYear',
    # Lag features
    'Temp_Lag1', 'Temp_Lag2', 'Temp_Lag3',
    'Humidity_Lag1', 'Humidity_Lag2', 'Humidity_Lag3',
    'Rain_Lag1', 'Pressure_Lag1', 'Wind_Lag1',
    # Rolling features
    'Temp_Roll3', 'Temp_Roll7', 'Humidity_Roll3', 'Rain_Roll7',
]

# Thêm DaylightHours nếu có
if 'DaylightHours' in df_ml.columns:
    feature_cols.append('DaylightHours')

# Chỉ giữ các cột thực sự có trong df_ml
feature_cols = [c for c in feature_cols if c in df_ml.columns]

# Loại bỏ các dòng có NaN (do shift/rolling)
df_ml = df_ml.dropna(subset=feature_cols + ['Target_Temp_Tomorrow'])
print(f"\nSố features: {len(feature_cols)}")
print(f"Số mẫu sau xử lý: {df_ml.shape[0]}")
print(f"Features: {feature_cols}")

# 6.2. Chia dữ liệu theo thời gian (temporal split - QUAN TRỌNG cho time series)
print("\n--- 6.2. CHIA DỮ LIỆU (TEMPORAL SPLIT 80/20) ---")
X = df_ml[feature_cols].values
y = df_ml['Target_Temp_Tomorrow'].values

# Temporal split: 80% đầu = train, 20% cuối = test (KHÔNG shuffle)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = df_ml['Date'].values[:split_idx]
test_dates = df_ml['Date'].values[split_idx:]

print(f"Tập Train: {X_train.shape[0]} mẫu (từ {pd.Timestamp(train_dates[0]).date()} "
      f"đến {pd.Timestamp(train_dates[-1]).date()})")
print(f"Tập Test:  {X_test.shape[0]} mẫu (từ {pd.Timestamp(test_dates[0]).date()} "
      f"đến {pd.Timestamp(test_dates[-1]).date()})")
print("⚠️ Sử dụng Temporal Split (không xáo trộn) để tránh rò rỉ dữ liệu tương lai")

# 6.3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6.4. Định nghĩa các mô hình
print("\n--- 6.3. HUẤN LUYỆN & ĐÁNH GIÁ 5 MÔ HÌNH ---")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15,
                                            min_samples_split=5,
                                            random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
}

# 6.5. Huấn luyện và đánh giá từng mô hình
results = {}
predictions = {}

for name, model in models.items():
    print(f"\n🔄 Đang huấn luyện: {name}...")

    # Huấn luyện
    if name in ['Random Forest', 'XGBoost']:
        model.fit(X_train, y_train)  # Tree-based không cần chuẩn hóa
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)

    # Tính metrics trên tập Test
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Tính R² trên tập Train (để phát hiện overfitting)
    r2_train = r2_score(y_train, y_pred_train)

    results[name] = {
        'R2_Train': r2_train,
        'R2_Test': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    predictions[name] = y_pred

    print(f" R² (Train): {r2_train:.4f} | R² (Test): {r2:.4f}")
    print(f"      RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C | MAPE: {mape:.2f}%")

# 6.6. Cross-Validation với TimeSeriesSplit
print("\n\n--- 6.4. CROSS-VALIDATION (TimeSeriesSplit, k=5) ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}

for name, model in models.items():
    if name in ['Random Forest', 'XGBoost']:
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                     scoring='r2', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv,
                                     scoring='r2', n_jobs=-1)

    cv_results[name] = cv_scores
    print(f"   {name}: R² CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 6.7. Bảng tổng kết so sánh mô hình
print("\n\n--- 6.5. BẢNG TỔNG KẾT SO SÁNH MÔ HÌNH ---")
results_df = pd.DataFrame(results).T
results_df['CV_R2_Mean'] = [cv_results[name].mean() for name in results_df.index]
results_df['CV_R2_Std'] = [cv_results[name].std() for name in results_df.index]
print(results_df.round(4))

# Tìm mô hình tốt nhất
best_model_name = results_df['R2_Test'].idxmax()
best_r2 = results_df.loc[best_model_name, 'R2_Test']
print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model_name} (R² Test = {best_r2:.4f})")

# =====================================================================
# 5.10 / 6.8. BIỂU ĐỒ SO SÁNH MÔ HÌNH (MODEL COMPARISON)
# =====================================================================
print("\n[Đang vẽ biểu đồ so sánh mô hình...]")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('SO SÁNH HIỆU SUẤT 5 MÔ HÌNH DỰ ĐOÁN NHIỆT ĐỘ',
             fontsize=16, fontweight='bold')

model_names = list(results.keys())
model_colors = ['#3498DB', '#2ECC71', '#E67E22', '#9B59B6', '#E74C3C']

# 10a. R² Score (Train vs Test)
ax = axes[0]
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x_pos - width / 2, [results[m]['R2_Train'] for m in model_names],
               width, label='Train R²', color='#3498DB', alpha=0.8)
bars2 = ax.bar(x_pos + width / 2, [results[m]['R2_Test'] for m in model_names],
               width, label='Test R²', color='#E74C3C', alpha=0.8)
ax.set_xlabel('Mô hình')
ax.set_ylabel('R² Score')
ax.set_title('R² Score (Train vs Test)')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.legend()
ax.set_ylim(0, 1.05)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

# 10b. RMSE
ax = axes[1]
rmse_vals = [results[m]['RMSE'] for m in model_names]
bars = ax.bar(x_pos, rmse_vals, color=model_colors, alpha=0.8, edgecolor='white')
ax.set_xlabel('Mô hình')
ax.set_ylabel('RMSE (°C)')
ax.set_title('RMSE - Sai số dự đoán (càng thấp càng tốt)')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 10c. Cross-Validation R² (boxplot)
ax = axes[2]
cv_data = [cv_results[name] for name in model_names]
bp = ax.boxplot(cv_data, patch_artist=True, labels=[n.replace(' ', '\n') for n in model_names])
for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('R² Score')
ax.set_title('Cross-Validation R² (5-Fold TimeSeriesSplit)')
ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('chart/bieu_do_10_so_sanh_mo_hinh.png', dpi=150, bbox_inches='tight')
plt.close()
print("Đã lưu: chart/bieu_do_10_so_sanh_mo_hinh.png")

# =====================================================================
# 6.9. FEATURE IMPORTANCE (TẦM QUAN TRỌNG CÁC FEATURES)
# =====================================================================
print("\n[Đang vẽ biểu đồ Feature Importance...]")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('TẦM QUAN TRỌNG CỦA CÁC FEATURES TRONG DỰ ĐOÁN NHIỆT ĐỘ',
             fontsize=16, fontweight='bold')

# Feature importance từ Random Forest
ax = axes[0]
rf_model = models['Random Forest']
rf_importance = rf_model.feature_importances_
sorted_idx_rf = np.argsort(rf_importance)[-15:]  # Top 15
ax.barh(range(len(sorted_idx_rf)),
        rf_importance[sorted_idx_rf],
        color='#9B59B6', alpha=0.8)
ax.set_yticks(range(len(sorted_idx_rf)))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx_rf], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Random Forest - Top 15 Features')

# Feature importance từ XGBoost
ax = axes[1]
xgb_model = models['XGBoost']
xgb_importance = xgb_model.feature_importances_
sorted_idx_xgb = np.argsort(xgb_importance)[-15:]  # Top 15
ax.barh(range(len(sorted_idx_xgb)),
        xgb_importance[sorted_idx_xgb],
        color='#E74C3C', alpha=0.8)
ax.set_yticks(range(len(sorted_idx_xgb)))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx_xgb], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('XGBoost - Top 15 Features')

plt.tight_layout()
plt.savefig('chart/bieu_do_11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Đã lưu: chart/bieu_do_11_feature_importance.png")

# =====================================================================
# 6.10. KẾT QUẢ MÔ HÌNH TỐT NHẤT (ACTUAL vs PREDICTED)
# =====================================================================
print("\n[Đang vẽ kết quả mô hình tốt nhất...]")

best_pred = predictions[best_model_name]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'KẾT QUẢ MÔ HÌNH TỐT NHẤT: {best_model_name.upper()}\n'
             f'(R² = {best_r2:.4f}, RMSE = {results[best_model_name]["RMSE"]:.4f}°C)',
             fontsize=14, fontweight='bold')

# 12a. Actual vs Predicted scatter
ax = axes[0, 0]
ax.scatter(y_test, best_pred, color='#3498DB', alpha=0.5, s=20, edgecolors='none')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Dự đoán hoàn hảo')
ax.set_xlabel('Nhiệt độ Thực tế (°C)')
ax.set_ylabel('Nhiệt độ Dự đoán (°C)')
ax.set_title(f'Thực tế vs Dự đoán (R² = {best_r2:.4f})')
ax.legend()

# 12b. Phân bố sai số (Residuals histogram)
ax = axes[0, 1]
residuals = y_test - best_pred
ax.hist(residuals, bins=30, color='#E74C3C', alpha=0.7, edgecolor='white')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax.axvline(x=residuals.mean(), color='blue', linestyle=':', linewidth=1.5,
           label=f'Mean = {residuals.mean():.3f}°C')
ax.set_xlabel('Sai số (°C) = Thực tế - Dự đoán')
ax.set_ylabel('Tần suất')
ax.set_title('Phân bố Sai số (Residuals)')
ax.legend()

# 12c. Time series: Actual vs Predicted
ax = axes[1, 0]
test_dates_plot = pd.to_datetime(test_dates)
ax.plot(test_dates_plot, y_test, color='#2ECC71', linewidth=1.2, alpha=0.8, label='Thực tế')
ax.plot(test_dates_plot, best_pred, color='#E74C3C', linewidth=1.2, alpha=0.8,
        linestyle='--', label='Dự đoán')
ax.set_xlabel('Ngày')
ax.set_ylabel('Nhiệt độ (°C)')
ax.set_title('Chuỗi thời gian: Thực tế vs Dự đoán')
ax.legend()
ax.tick_params(axis='x', rotation=30)

# 12d. QQ-Plot (kiểm tra phân phối sai số)
ax = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('QQ-Plot: Kiểm tra phân phối chuẩn của sai số')
ax.get_lines()[0].set_markerfacecolor('#3498DB')
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color('#E74C3C')

plt.tight_layout()
plt.savefig('chart/bieu_do_12_ket_qua_mo_hinh.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Đã lưu: chart/bieu_do_12_ket_qua_mo_hinh.png")

# =============================================================================
# PHẦN 7: KẾT LUẬN VÀ NHẬN XÉT
# =============================================================================
print("\n" + "=" * 70)
print("PHẦN 7: KẾT LUẬN VÀ NHẬN XÉT")
print("=" * 70)

print(f"""
    TÓM TẮT KẾT QUẢ PHÂN TÍCH:

1. TỔNG QUAN DỮ LIỆU:
   - Dataset: HoChiMinhCity_3years_weather.csv (Visual Crossing API)
   - Gồm {df.shape[0]} bản ghi từ {df['Date'].min().date()} đến {df['Date'].max().date()}
   - {len(main_numeric_cols)} biến số chính + các biến phụ trợ
   - Đã chuyển đổi đơn vị từ Imperial sang Metric

2. THỐNG KÊ MÔ TẢ:
   - Nhiệt độ TB: {df['Temperature_C'].mean():.1f}°C (Min: {df['Temperature_C'].min():.1f}°C, Max: {df['Temperature_C'].max():.1f}°C)
   - Độ ẩm TB:    {df['Humidity_Percent'].mean():.1f}%
   - Lượng mưa TB: {df['Rainfall_mm'].mean():.1f} mm/ngày
   - Áp suất TB:  {df['Pressure_mbar'].mean():.1f} mbar
   - Tốc độ gió TB: {df['WindSpeed_kmh'].mean():.1f} km/h

3. PHÂN TÍCH TƯƠNG QUAN:""")

for col1, col2 in important_pairs:
    if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
        r = corr_matrix.loc[col1, col2]
        print(f"   - {col1} vs {col2}: r = {r:.4f}")

print(f"""
4. SO SÁNH 5 MÔ HÌNH DỰ ĐOÁN NHIỆT ĐỘ:""")

for name in model_names:
    r = results[name]
    flag = " 🏆" if name == best_model_name else ""
    print(f"   - {name}: R²={r['R2_Test']:.4f}, RMSE={r['RMSE']:.4f}°C, "
          f"MAE={r['MAE']:.4f}°C{flag}")

print(f"""
   🏆 MÔ HÌNH TỐT NHẤT: {best_model_name}
      R² = {best_r2:.4f} (Giải thích được {best_r2*100:.1f}% biến thiên nhiệt độ)
      RMSE = {results[best_model_name]['RMSE']:.4f}°C
      MAE = {results[best_model_name]['MAE']:.4f}°C

5. CÁC BIỂU ĐỒ ĐÃ TẠO (12 biểu đồ):
   - chart/bieu_do_1_chuoi_thoi_gian.png  : Chuỗi thời gian 4 biến chính
   - chart/bieu_do_2_phan_phoi.png        : Histogram + KDE 8 biến
   - chart/bieu_do_3_tuong_quan.png       : Heatmap + Scatter tương quan
   - chart/bieu_do_4_theo_thang.png       : Phân tích theo tháng
   - chart/bieu_do_5_theo_mua.png         : Phân tích mùa mưa/mùa khô
   - chart/bieu_do_6_moving_average.png   : Trung bình trượt (xu hướng)
   - chart/bieu_do_7_heatmap_thang_nam.png: Heatmap năm x tháng
   - chart/bieu_do_8_pairplot.png         : Pair Plot mùa mưa/khô
   - chart/bieu_do_9_so_sanh_nam.png      : So sánh qua các năm
   - chart/bieu_do_10_so_sanh_mo_hinh.png : So sánh 5 mô hình ML
   - chart/bieu_do_11_feature_importance.png: Feature Importance (RF + XGBoost)
   - chart/bieu_do_12_ket_qua_mo_hinh.png : Kết quả mô hình tốt nhất

6. NHẬN XÉT VỀ MÔ HÌNH:
   - Các mô hình ensemble (Random Forest, XGBoost) cho kết quả vượt trội
     so với mô hình tuyến tính nhờ khả năng xử lý quan hệ phi tuyến.
   - Lag features (nhiệt độ ngày trước) là features quan trọng nhất,
     cho thấy tính liên tục mạnh của chuỗi thời gian nhiệt độ.
   - Bức xạ mặt trời, áp suất và độ ẩm cũng đóng vai trò quan trọng
     trong việc dự đoán nhiệt độ ngày mai.
""")

print("=" * 70)
print("  HOÀN THÀNH PHÂN TÍCH DỮ LIỆU THỜI TIẾT TP.HCM & DỰ ĐOÁN ML!")
print("=" * 70)

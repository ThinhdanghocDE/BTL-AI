import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
from datetime import datetime

# --- CẤU HÌNH TRANG (ĐẶT LÊN ĐẦU TIÊN) ---
st.set_page_config(page_title="Dự đoán Gửi tiền Ngân hàng", layout="wide")

# --- TẢI CÁC THÀNH PHẦN ĐÃ LƯU ---
# Sử dụng cache để không phải tải lại mô hình và scaler mỗi lần tương tác
@st.cache_resource
def load_model():
    """Tải mô hình Keras đã được huấn luyện."""
    try:
        model = tf.keras.models.load_model('C:/Users/ADMIN/Documents/AI/AI&ML (2)/model/final_model.h5')
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

@st.cache_resource
def load_scalers_and_info():
    """Tải các scaler và thông tin cột."""
    try:
        robust_scaler = joblib.load('C:/Users/ADMIN/Documents/AI/AI&ML (2)/model/robust_scaler.pkl')
        min_max_scaler = joblib.load('C:/Users/ADMIN/Documents/AI/AI&ML (2)/model/min_max_scaler.pkl')
        with open('C:/Users/ADMIN/Documents/AI/AI&ML (2)/model/final_columns.json', 'r') as f:
            final_columns = json.load(f)
        with open('C:/Users/ADMIN/Documents/AI/AI&ML (2)/model/age_group_map.json', 'r') as f:
            age_group_map = json.load(f)
        return robust_scaler, min_max_scaler, final_columns, age_group_map
    except Exception as e:
        st.error(f"Lỗi khi tải các thành phần tiền xử lý: {e}")
        return None, None, None, None

# Tải tất cả các thành phần cần thiết
model = load_model()
robust_scaler, min_max_scaler, final_columns, age_group_map = load_scalers_and_info()


# --- GIAO DIỆN NGƯỜI DÙNG STREAMLIT ---
st.title("️🏦 Ứng dụng Dự đoán Khả năng Gửi tiền của Khách hàng")
st.write("Cung cấp thông tin của khách hàng để mô hình dự đoán khả năng họ sẽ đăng ký một khoản tiền gửi có kỳ hạn.")

# Tạo các cột để bố trí giao diện
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Thông tin cá nhân")
    age = st.number_input("Tuổi", min_value=18, max_value=100, value=40)
    job_options = ['management', 'technician', 'blue-collar', 'admin.', 'retired', 'No Income', 'Unstable Income']
    job = st.selectbox("Nghề nghiệp", options=job_options)
    marital = st.selectbox("Tình trạng hôn nhân", options=['married', 'single', 'divorced'])
    education = st.selectbox("Học vấn", options=['tertiary', 'secondary', 'primary', 'unknown'])

with col2:
    st.header("Thông tin tài chính liên hệ")
    balance = st.number_input("Số dư tài khoản (Dolla)", value=1000)
    default = st.selectbox("Có nợ xấu không?", options=['no', 'yes'])
    housing = st.selectbox("Có vay mua nhà không?", options=['yes', 'no'])
    loan = st.selectbox("Có vay cá nhân không?", options=['yes', 'no'])
    contact = st.selectbox("Phương thức liên hệ", options=['cellular', 'telephone', 'unknown'])


with col3:
    st.header("Thông tin chiến dịch")
    duration = st.number_input("Thời gian cuộc gọi", min_value=0, value=300)
    campaign = st.number_input("Số lần liên hệ trong chiến dịch này", min_value=1, value=2)
    month_name = st.selectbox("Tháng liên hệ cuối cùng",
                               options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    pdays = st.number_input("Số ngày kể từ lần liên hệ cuối cùng (nhập -1 nếu chưa liên hệ)", min_value=-1, value=-1)
    previous = st.number_input("Số lần liên hệ trước chiến dịch này", min_value=0, value=0)
    poutcome = st.selectbox("Kết quả của chiến dịch trước", options=['success', 'failure', 'other', 'unknown'])

# Nút dự đoán
if st.button("Dự đoán 🚀", use_container_width=True):
    if model and robust_scaler and min_max_scaler and final_columns and age_group_map:
        # --- BƯỚC 1: TẠO DATAFRAME TỪ DỮ LIỆU ĐẦU VÀO ---
        input_data = {
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [datetime.strptime(month_name, "%b").month],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
        }
        input_df = pd.DataFrame(input_data)

        # Thêm cột tuổi gốc để xử lý
        input_df['age'] = [age]

        # --- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (Y HỆT NOTEBOOK) ---

        # 2.1. Xử lý nhóm tuổi (age_group)
        bins = [18, 30, 40, 50, 100]
        labels = ['18-30', '31-40', '41-50', '51-61']
        input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False).astype(str)
        input_df.drop('age', axis=1, inplace=True)

        # 2.2. Áp dụng Target Encoding cho age_group
        input_df['age_group'] = input_df['age_group'].map(age_group_map)

        # 2.3. Chuẩn hóa các cột số
        input_df['balance'] = robust_scaler.transform(input_df[['balance']])
        numeric_cols_to_scale = ['duration', 'campaign', 'pdays', 'previous']
        input_df[numeric_cols_to_scale] = min_max_scaler.transform(input_df[numeric_cols_to_scale])
        input_df['pdays'] = input_df['pdays'].replace(-1, 0) # Xử lý pdays = -1

        # 2.4. Mã hóa cột nhị phân
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            input_df[col] = input_df[col].map({'yes': 1, 'no': 0})

        # 2.5. Áp dụng One-Hot Encoding
        input_df_encoded = pd.get_dummies(input_df)

        # 2.6. Sắp xếp lại các cột để khớp với dữ liệu huấn luyện
        # Tạo một DataFrame trống với đúng các cột
        final_df = pd.DataFrame(columns=final_columns)
        # Điền dữ liệu từ input đã mã hóa vào
        final_df = pd.concat([final_df, input_df_encoded], ignore_index=True).fillna(0)
        # Đảm bảo thứ tự cột chính xác
        final_df = final_df[final_columns]

        # --- BƯỚC 3: DỰ ĐOÁN ---
        prediction_proba = model.predict(final_df)
        prediction = (prediction_proba > 0.4).astype(int) # Sử dụng ngưỡng tối ưu của bạn

        # --- BƯỚC 4: HIỂN THỊ KẾT QUẢ ---
        st.subheader("Kết quả Dự đoán")
        probability = prediction_proba[0][0]

        if prediction[0] == 1:
            st.success(f"**Khách hàng có khả năng sẽ gửi tiền.** (Xác suất: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"**Khách hàng có khả năng sẽ KHÔNG gửi tiền.** (Xác suất không gửi: {1-probability:.2%})")
        with st.expander("Xem chi tiết dữ liệu đã xử lý"):
            st.dataframe(final_df)
    else:
        st.error("Mô hình hoặc các thành phần chưa được tải. Vui lòng kiểm tra lại file.")
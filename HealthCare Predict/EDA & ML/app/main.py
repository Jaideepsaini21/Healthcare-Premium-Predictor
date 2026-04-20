# if st.button("Predict"):
#     pred = predict(input_dict)
#     st.success(f"Predicted Premium: {pred}")
#     print(input_dict)

import os
import sqlite3

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def get_db_path():
    return os.path.join(APP_DIR, "data", "user_data.db")
    # custom_path = os.getenv("HEALTHCARE_DB_PATH")
    # if custom_path:
    #     return os.path.abspath(os.path.expandvars(os.path.expanduser(custom_path)))

    
    # local_app_data = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    # return os.path.join(local_app_data, "HealthcarePremiumPredictor", "user_data.db")


DB_PATH = get_db_path()
DB_DIR = os.path.dirname(DB_PATH)


def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)

    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                
                income REAL,
                
                insurance_plan TEXT,
                employment TEXT,
                gender TEXT,
                marital_status TEXT,
                bmi TEXT,
                smoking TEXT,
                region TEXT,
                medical_history TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("PRAGMA journal_mode = WAL")


init_db()


def save_to_db(data):
    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO user_data (
                age,
                income,
                insurance_plan,
                employment,
                gender,
                marital_status,
                bmi,
                smoking,
                region,
                medical_history
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data['Age'],
                # data['Number of Dependants'],
                data['Income in Lakhs'],
                # data['Genetical Risk'],
                data['Insurance Plan'],
                data['Employment Status'],
                data['Gender'],
                data['Marital Status'],
                data['BMI Category'],
                data['Smoking Status'],
                data['Region'],
                data['Medical History'],
            ),
        )

    return {"id": cursor.lastrowid}


import streamlit as st
from predict import predict
import matplotlib.pyplot as plt

# ✅ Get logo path (safe for any system)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
logo_path = os.path.join(BASE_DIR, "datasets", "VW-Logo-and-Tagline.png")

# ✅ Show logo on left side
col1, col2 = st.columns([3, 7])

with col1:
    st.image(logo_path, width=500)

with col2:
    st.empty()  # blank space (no header text)
    
# st.image("../datasets/Virtuowhiz.jpeg", width='stretch')
# -------------------- CONFIG --------------------
st.set_page_config(page_title="Healthcare Premium Predictor", layout="wide")

# -------------------- HEADER --------------------
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

with col3:
    st.link_button("🔗 LinkedIn", "https://www.linkedin.com/company/virtuowhiz/posts/?feedView=all", use_container_width=True)
with col4:
    st.link_button("🌐 Website", "https://virtuowhiz.com/", use_container_width=True)
    
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🏥 Healthcare Premium Predictor</h1>
<p style='text-align: center;'>AI-based smart insurance pricing system</p>
""", unsafe_allow_html=True)

st.divider()

# -------------------- OPTIONS --------------------
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer'],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease',
        'Diabetes & Thyroid', 'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# -------------------- VARIABLES --------------------
pred, yearly, monthly = None, None, None

# -------------------- LAYOUT --------------------
left, right = st.columns([2, 1])

# ================= LEFT SIDE =================
with left:

    st.subheader("🧍 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        # age = st.number_input('Age', 18, 100)
        age = st.number_input('Age ✱', min_value=18, max_value=100, value=None, placeholder="Enter Age")
    with col2:
        gender = st.selectbox('Gender', ["Select"] + categorical_options['Gender'])
    with col3:
        marital_status = st.selectbox('Marital Status', ["Select"] + categorical_options['Marital Status'])

    st.subheader("🏥 Health Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        bmi_category = st.selectbox('BMI ✱', ["Select"] + categorical_options['BMI Category'])
    with col5:
        smoking_status = st.selectbox('Smoking ✱', ["Select"] + categorical_options['Smoking Status'])
    with col6:
        # genetical_risk = st.number_input('Genetic Risk', 0, 5)
        medical_history = st.selectbox('Medical History ✱', ["Select"] + categorical_options['Medical History'])

    colA, colB = st.columns(2)
    # with colA:
    #     medical_history = st.selectbox('Medical History ✱', ["Select"] + categorical_options['Medical History'])
    # with colB:
    #     number_of_dependants = st.number_input('Dependants', 0, 10)

    st.subheader("💼 Financial Details")
    col9, col10, col11 = st.columns(3)
    with col9:
        income_lakhs = st.number_input('Income (Lakhs) ✱', 0, 200)
    with col10:
        employment_status = st.selectbox('Employment', ["Select"] + categorical_options['Employment Status'])
    with col11:
        insurance_plan = st.selectbox('Plan ✱', ["Select"] + categorical_options['Insurance Plan'])

    region = st.selectbox('Region', ["Select"] + categorical_options['Region'])

# ================= RIGHT SIDE =================
def validate_inputs():
    required_fields = [
        age,
        bmi_category,
        smoking_status,
        medical_history,
        income_lakhs,
        insurance_plan
    ]

    # Check empty or "Select"
    for field in required_fields:
        if field is None or field == "Select":
            return False

    return True

with right:
    st.subheader("📊 Prediction Result")

    if st.button("🚀 Predict Premium", use_container_width=True, key="predict_btn"):
        if not validate_inputs():
            st.warning("⚠️ Please fill all required fields")
            st.stop()
            
        input_dict = {
            'Age': age,
            # 'Number of Dependants': number_of_dependants,  # removed from UI
            'Income in Lakhs': income_lakhs,
            # 'Genetical Risk': genetical_risk,
            'Insurance Plan': insurance_plan,
            'Employment Status': employment_status,
            'Gender': gender,
            'Marital Status': marital_status,
            'BMI Category': bmi_category,
            'Smoking Status': smoking_status,
            'Region': region,
            'Medical History': medical_history
        }

        pred = predict(input_dict)
        yearly = pred
        monthly = round(pred / 12, 2)
        save_to_db(input_dict)
        
        # 💰 Premium Card
        st.markdown(f"""
        <div style="background:#D4EFDF;padding:20px;border-radius:10px;text-align:center;">
        <h3>💰 Premium</h3>
        <h2>₹ {yearly} / year</h2>
        <p>₹ {monthly} / month</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(yearly / 20000 * 100), 100))

        # # 📊 Chart
        # fig, ax = plt.subplots()
        # ax.bar(['Monthly', 'Yearly'], [monthly, yearly])
        # ax.set_title("Premium Comparison")
        # st.pyplot(fig)

        # 📄 PDF
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        def create_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()
            content = []
            content.append(Paragraph("Healthcare Premium Report", styles['Title']))
            content.append(Paragraph(f"Yearly: ₹ {yearly}", styles['Normal']))
            content.append(Paragraph(f"Monthly: ₹ {monthly}", styles['Normal']))
            doc.build(content)
            return "report.pdf"

        file = create_pdf()
        with open(file, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")
            
        with st.expander("🧾 View Entered Details"):
                for key, value in input_dict.items():
                    st.write(f"**{key}:** {value}")
        print(input_dict)

# ================= DASHBOARD LEFT =================
if pred is not None:
    with left:
        st.divider()
        st.subheader("📊 Advanced Insights")

        # Risk
        if yearly < 10000:
            risk = "Low 🟢"
        elif yearly < 20000:
            risk = "Medium 🟡"
        else:
            risk = "High 🔴"

        c1, c2, c3 = st.columns(3)
        c1.metric("Yearly", f"₹ {yearly}")
        c2.metric("Monthly", f"₹ {monthly}")
        c3.metric("Risk", risk)

        st.markdown("### 📊 Market Comparison")
        market_avg = 10000

        c4, c5 = st.columns(2)
        c4.metric("Your", f"₹ {yearly}")
        c5.metric("Market", f"₹ {market_avg}")

        diff = yearly - market_avg
        if diff > 0:
            st.warning(f"₹ {diff} higher than market")
        else:
            st.success(f"₹ {abs(diff)} cheaper than market")
                    
                    
                                 
                    
                    
                    
                    
                    
                    
                    
                    
                    

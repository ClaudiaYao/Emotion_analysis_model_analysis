import pandas as pd
import common_code
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn import metrics

st.title("🤪 Emotion Analysis - Model Demo")
st.write("""
The task is to predict the emotions expressed in short text documents.
""")

# Start a form in the sidebar
with st.sidebar.form(key='my_form'):
    st.write("TfidfVectorizer parameters:")
    slider_max_tokens = st.slider(label='max_tokens', min_value=5000, max_value=20000, value=10000, step= 100)
    select_ngram_lower_range = st.selectbox(label='ngram_range_lower', options = [i for i in range(1, 3)], index = 0, disabled=True)
    select_ngram_upper_range = st.selectbox(label='ngram_range_upper', options=[i for i in range(1, 6)], disabled=False, index = 0)

    st.write("LogisticRegression parameters:")
    c_value = st.slider(label='C', min_value=0.5, max_value=2.0, value=0.85, step = 0.05)
    select_solver = st.selectbox(label='solver', options = ["newton-cg", "sag", "saga", "lbfgs"], index = 2, disabled=False)
    submitted = st.form_submit_button('Submit')

if submitted:
    st.write("submitted new")
    MAX_FEATURES = slider_max_tokens
    NGRAMS_RANGE = (select_ngram_lower_range, select_ngram_upper_range)
    SOLVER = select_solver
    C_VALUE = c_value
else:
    MAX_FEATURES = 10000
    NGRAMS_RANGE = (1, 1)
    SOLVER = "saga"
    C_VALUE = 0.85

# Part 1 - load cleaned training data
USE_EXISTING_CLEANED_FILES = True
if st.button("Re-clean the training data", type="primary"):
    with st.spinner(text="In progress..."):
        df_train = pd.read_csv("data/train.csv")
        df_test = pd.read_csv("data/test.csv")
        df_train, df_test = common_code.clean_text_wrapper(df_train, df_test)
    st.info('Finish re-clean the training data.', icon="ℹ️")


if USE_EXISTING_CLEANED_FILES:
    df_train = pd.read_csv("data/cleaned_train.csv", encoding="utf-8")
    df_train['cleaned_text'].replace(np.nan, "NA", inplace=True)
    df_test = pd.read_csv("data/cleaned_test.csv", encoding = "utf-8")
    df_test['cleaned_text'].replace(np.nan, "NA", inplace=True)


st.write('## Training Data')
st.write("#### 🚀 Cleaned Training Data")
st.write(df_train)

st.write("#### 📊 Emotion Type Distribution")
col1, col2 = st.columns([2, 1])
with col1:
    fig = common_code.draw_piechart(df_train)
    st.pyplot(fig)

le_name_mapping, le_label_mapping, y_label = common_code.encode_label(df_train)

# Part 2 - run model
text_train, text_test = common_code.encode_doc(MAX_FEATURES, NGRAMS_RANGE, df_train['cleaned_text'], df_test['cleaned_text'])

# split the training dataset for training and verification
indices = np.arange(df_train.shape[0])
X_train, X_veri, y_train, y_veri, indices_train, indices_veri = train_test_split(
    text_train, y_label, indices, test_size=0.1, random_state=42)

st.markdown("""
**📌 After splitting the training data,**\n
**Training data X-shape, Y-shape:**\t\t{}, {}\n
**Verification data X-shape, Y-shape:**\t\t{}, {}
        """.format(X_train.shape, y_train.shape, X_veri.shape, y_veri.shape))

st.header("🚴🏿‍♂️ Run LogisticRegression model...")
# create the model and fit, predict
log_model = LogisticRegression(C=C_VALUE, solver=SOLVER, max_iter=1000)
log_model.fit(X_train, y_train)
y_veri_pred = log_model.predict(X_veri)

st.subheader("✍️ Result")
col1, col2, _, _ = st.columns(4)
with col1:
    st.write("**Methic**")
    st.write("Accuracy")
    st.write("F1 Score - Micro")
    st.write("F1 Score - Macro")

with col2:
    st.write("**Value**")
    st.write(metrics.accuracy_score(y_veri, y_veri_pred))
    st.write(round(metrics.f1_score(y_veri, y_veri_pred, average='micro'),6))
    st.write(round(metrics.f1_score(y_veri, y_veri_pred, average='macro'), 6))


st.write("**Confusion Matrix:**\n")
col1, col2 = st.columns([2, 1])
with col1:
    cm = metrics.confusion_matrix(y_veri, y_veri_pred)
    fig = common_code.draw_confusion_matrix(cm, [le_label_mapping[i] for i in range(5)])

    # st.image(disp.confusion_matrix)
    st.pyplot(fig)






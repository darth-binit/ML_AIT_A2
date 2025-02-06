import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from streamlit_option_menu import option_menu
from streamlit.components.v1 import components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import os
from model.model import MyRegression, LassoRegression, L1Penalty, plot_feature_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

##path names
brand_means_path = os.path.join(BASE_DIR, "brand_means.pkl")
pipeline_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler1.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
model_L_path = os.path.join(BASE_DIR, "myregression_lasso.pkl")
css_path = css_path = os.path.join(BASE_DIR, "style", "style.css")

##loading the model
brand_means = joblib.load(brand_means_path)
best_pipeline = joblib.load(pipeline_path)
encoder = joblib.load(encoder_path)
scaler_modelA2 = joblib.load(scaler_path)
model_l = joblib.load(model_L_path)


data_file_path = os.path.join(BASE_DIR, "Out_287.csv")
df = pd.read_csv(data_file_path)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

## types of column

cat_col = df.select_dtypes(include='object').columns.tolist()
num_col = df.select_dtypes(include='number').columns.tolist()

##function for prediction
def predict_A1(val):
    print('A1_input: ',val)
    val['brand_encoded'] = val['brand'].map(brand_means)
    val['brand_encoded'] = val['brand_encoded'].fillna(brand_means.mean())
    val = val.drop(columns=['brand'])

    prediction_log = best_pipeline.predict(val)
    prediction = np.expm1(prediction_log)

    return prediction

def predict_A2(data, return_preprocessed=False):
    # Define Feature Groups
    categorical_features = ['fuel', 'seller_type', 'transmission']
    numerical_features = ['year', 'km_driven', 'owner', 'mileage', 'engine', 'max_power', 'seats']

    # 1️⃣ **Apply Brand Encoding First (Before Dropping `brand`)**
    data['brand_encoded'] = data['brand'].map(brand_means)

    # Handle unseen brands
    if data['brand_encoded'].isnull().any():
        data['brand_encoded'] = brand_means.mean()

    data.drop(columns=['brand'], inplace=True)
    data_encoded = encoder.transform(data[categorical_features])
    data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(categorical_features))
    data.drop(columns=categorical_features, inplace=True)
    data = pd.concat([data.reset_index(drop=True), data_encoded_df.reset_index(drop=True)], axis=1)
    final_numerical_features = numerical_features + ['brand_encoded']
    data[final_numerical_features] = scaler_modelA2.transform(data[final_numerical_features])
    data_final = data.values
    predicted_price = np.expm1(model_l._predict(data_final))

    column_names = data.columns.tolist()

    if return_preprocessed:
        return column_names

    return predicted_price


st.set_page_config(layout="wide")


def streamlit_menu():


    st.markdown(
        "<h1 style='text-align: center; color: #355F10;'>Chaky Automobiles Solution</h1>",
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,  # required
        options=["Home | Descriptive ", "Predictive Analytics"],  # required
        icons=["bi bi-activity", "bi bi-clipboard-data", "bi bi-pie-chart"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#C6DDB6", },
            "icon": {"color": "#072810", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",

                "padding":"10px"},
            "nav-link-selected": {"background-color": "#355F10"},
                    },)
    return selected

select  = streamlit_menu()

if select == "Home | Descriptive ":
    col1, col2, col3 = st.columns([1,0.13,1])
    col1_bg_color = "#c1d6c1"
    col2_bg_color = "#e4f2e4"

    with col2:
        icon_url = os.path.join(BASE_DIR, "auto-automobile-car-pictogram-service-traffic-transport--2.png")
        st.image(icon_url, width=50)

    with col1:
        st.markdown(
            f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;">Numerical Features Distribution</div>',
            unsafe_allow_html=True)
        feat_select = st.selectbox('Select Feature', df[num_col].columns)

        fig = px.histogram(
            df,
            x=feat_select,
            nbins=30,
            marginal="violin",  # Add KDE (marginal distribution)
            histnorm="density",
            opacity=0.75,
            color_discrete_sequence=['#379037'],
        )

        fig.update_layout(
            xaxis_title=feat_select,
            yaxis_title="Density",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)


    with col3:
        st.markdown(
            f'<div style = "background-image: linear-gradient(to left, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;">Categorical Features Count</div>',
            unsafe_allow_html=True)
        cat_feat_select = st.selectbox('Select Feature', df[cat_col].columns)

        colors = ['#379037', '#71B971', '#9DD39D']
        feat_val = df[cat_feat_select].value_counts().values
        total = sum(feat_val)
        percentages = [f'{(v / total) * 100:.2f}%' for v in feat_val]

        trace = go.Bar(x=df[cat_feat_select].value_counts().index,
                       y=df[cat_feat_select].value_counts().values, marker=dict(color=colors),
                       hovertext=percentages)
        layout = go.Layout()
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    ## Correlation Plot
    st.markdown('\n\n')
    st.markdown(
        f'<div style="background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Correlation | Numerical Features </div>',
        unsafe_allow_html=True)
    st.markdown('\n\n\n\n')

    fig, ax = plt.subplots(figsize=(30, 16))
    num_corr = df[num_col].corr()
    mask = np.triu(np.ones_like(num_corr, dtype=bool))
    sns.heatmap(num_corr, mask=mask, xticklabels=num_corr.columns, yticklabels=num_corr.columns, annot=True, linewidths=.3,
                     cmap='Greens', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    ## Feature importance bar plot
    st.markdown('\n\n')
    st.markdown(
        f'<div style="background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Correlation | Feature Importance_ </div>',
        unsafe_allow_html=True)
    st.markdown('\n\n\n\n')

    fig = plot_feature_importance(model_l._coef() , predict_A2(df.drop(columns=['selling_price']), return_preprocessed=True))

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


if select == 'Predictive Analytics':
    colp1, colp2 = st.columns([1,1])

    with open(css_path) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    with colp2:
        prediction_placeholder = st.empty()
        st.session_state['prediction_result'] = ''
        prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                        <h2 style='text-align: center'>Prediction</h2>
                                                        <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}</p></div>""",
                                        unsafe_allow_html=True)

        st.markdown('\n\n')

        if "model_selected" not in st.session_state:
            st.session_state["model_selected"] = None

        if "predict_clicked" not in st.session_state:
            st.session_state["predict_clicked"] = False

        # **Model Selection**
        colb1, colb2 = st.columns(2)

        with colb1:
            if st.button("🌲 Random Forest (Model A1)", key="model_A1"):
                st.session_state["model_selected"] = "Random Forest Model A1"
                st.session_state["predict_clicked"] = True

        with colb2:
            if st.button("📈 Regression (Model A2)", key="model_A2"):
                st.session_state["model_selected"] = "Regression Model A2"
                st.session_state["predict_clicked"] = True

        st.success(f"✅ {'&nbsp;' * 20} Selected Model {'&nbsp;' * 2} | {'&nbsp;' * 10}  💡{st.session_state['model_selected']}")

        # predict_btn =  st.button("Predict", st124783="predict", use_container_width=True)
        brand = st.selectbox('Brand', df['brand'].unique())

        colpp1, colpp2, colpp3 = st.columns([1,1,1])
        with colpp1:
            year = st.selectbox('Year', sorted(df['year'].unique()))
            seats = st.selectbox('Seats', sorted(df['seats'].unique()))
            fuel = st.selectbox('Fuel Type', df['fuel'].unique())
        with colpp2:
            seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
            mileage = st.number_input('Mileage', min_value=5, max_value=42, value=15)
            kms_driven = st.number_input('Kilometer Driven', min_value = 1000, max_value=2500000, value=1000)
        with colpp3:
            owner = st.number_input('Ownership', min_value=1, max_value=4, value=1)
            engine = st.number_input('Engine', min_value=500, max_value=4000, value=1400)
            max_power = st.number_input('Maximum Power', min_value=30, max_value=400, value = 85)

        transmission = st.selectbox('Transmission', df['transmission'].unique())

    usr_data = {'brand': [brand], 'year': [year], 'km_driven': [kms_driven], 'fuel': [fuel], 'seller_type': [seller_type], 'transmission': [transmission], 'owner': [owner], 'mileage': [mileage],
                'engine': [engine], 'max_power': [max_power], 'seats': [seats]}
    usr_data = pd.DataFrame(usr_data)

    with colp1:
        sel_cols = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
                                        'mileage', 'engine', 'max_power', 'seats']
        fuel_map = {'Diesel': 2, 'Petrol': 1}
        seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
        transmission_map = {'Automatic': 2, 'Manual': 1}

        temp_df = df.drop(columns=['brand','selling_price'])
        temp_df = temp_df.replace(fuel_map)
        temp_df = temp_df.replace(seller_map)
        temp_df = temp_df.replace(transmission_map)

        scaler = MinMaxScaler()
        scaled = scaler.fit(temp_df)

        mod_usr_data = usr_data.replace(fuel_map)
        mod_usr_data = mod_usr_data.replace(seller_map)
        mod_usr_data = mod_usr_data.replace(transmission_map)
        mod_usr_data = mod_usr_data.drop(columns=['brand'])


        scaled_data = scaler.transform(mod_usr_data[sel_cols])

        scaled_df = pd.DataFrame(scaled_data, columns=sel_cols)

        fig = px.line_polar(scaled_df, scaled_df.values.reshape(-1), theta=sel_cols, line_close=True)
        fig.update_traces(fill='toself', line_color='green')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('\n\n\n')
        st.markdown(f"""<div style='background-color:#c1d6c1 ; padding: 60px; border-radius: 3px'>
                                                        <h2 style='text-align: center'>Notes</h2>
                                                        <p style='font-size: 13px; text-align: center'> This is new and robust <strong>Version A2 model</strong>, which now features two models A1 and A2, this is built
                                                             entirely by using custom classes to create Regression Models from scatch, <strong>(Vanilla Regression | Ridge | Lasso)</strong>. The developer with rigourous testing 
                                                             found out that regularization works well to stablize the model thereby giving good results. For using the model, it is similar to previous version
                                                              only that here you have to <span style='color:#C70039; font-weight:bold;'>press the button of the model</span> of your choice and then the model will predict an output for you.
                                                              The author have choosen Regression with L1 penalty (Lasso).Since we have a discrete and a continuos feature, ensemble/trees model perform better regression.</p></div>""",

                                        unsafe_allow_html=True)

    if st.session_state["predict_clicked"]:

        if st.session_state["model_selected"]:

            if st.session_state["model_selected"] == "Random Forest Model A1":# Call the respective model function
                pred_val = predict_A1(usr_data)[0]
            elif st.session_state["model_selected"] == "Regression Model A2":
                pred_val = predict_A2(usr_data)[0]
            else:
                st.error("⚠️ Invalid Model Selection!")
                st.stop()

            # Process and display prediction
            if pred_val is not None:
                pred_val = np.round(pred_val, decimals=2)
                formatted_val = f"{pred_val:,.2f}"
                st.session_state['prediction_result'] = formatted_val
                print("Predicted Value:", formatted_val)
            else:
                formatted_val = "Not Available"

            # Display Prediction
            prediction_placeholder.markdown(f"""
                <div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                    <h2 style='text-align: center'>Prediction | Price</h2>
                    <p style='font-size: 24px; text-align: center'>{formatted_val}</p>
                </div>""",
                                            unsafe_allow_html=True)

            # Reset "predict_clicked" state after prediction is done
            st.session_state["predict_clicked"] = False







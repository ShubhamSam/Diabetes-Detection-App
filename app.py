import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a title and sub-title
st.write("""
# Diabetes Detection
 Detect if someone has diabetes using machine learning and python !
""")

# open and diaplay the image
image = Image.open(
    'C:/Users/Shubham/Desktop/My Projects/Diabetese App/diabetese.png')
st.image(image, caption='ML', use_column_width=True)

# get the data
df = pd.read_csv(
    'C:/Users/Shubham/Desktop/My Projects/Diabetese App/diabetes_datasets.csv')

# SET A SUB HEADER
st.subheader('Data Information: ')

# show the data as a table
st.dataframe(df)

# show statistics on the data
st.write(df.describe())

# show the data as a chart
chart = st.bar_chart(df)

# split the data into independent x and dependent y variable
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# split the data into 75% Training and 25% Testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

# get the feature input from the user


def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age
    }

    # transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# store the user imput into a variable
user_input = get_user_input()

# set a subheader and display the user input
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# show the models matrics
st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(y_test, model.predict(x_test)) * 100) + '%')


# store the model predictions in a variable
prediction = model.predict(user_input)

# set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)

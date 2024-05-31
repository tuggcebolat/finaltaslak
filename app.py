import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
from math import log

# Flask uygulamasını oluştur
app = Flask(__name__)

#Modeli yükle
model_path = os.path.join("templates", "model.pkl")
model = pickle.load(open(model_path, "rb"))


UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Dosya yükleme klasörünü oluştur
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def create_histograms_and_violin_plots(data, num_columns):
    histograms = []
    violin_plots = []

    for col in num_columns:
        # Histogram
        histogram = go.Histogram(
            x=data[col],
            name=col,
            marker=dict(color='rgb(255, 102, 102)'),  # Kırmızı renk
        )
        histograms.append(histogram)

        # Violin plot
        violin_plot = go.Violin(
            y=data[col],
            name=col,
            marker=dict(color='rgb(255, 102, 102)'),  # Kırmızı renk
            box_visible=True
        )
        violin_plots.append(violin_plot)

    return histograms, violin_plots

def create_categorical_plots(data, cat_columns):
    cat_plots = {}

    for col in cat_columns:
        data_col = data[col].value_counts().reset_index()
        data_col.columns = [col, 'count']

        trace = go.Bar(
            x=data_col[col],
            y=data_col['count'],
            marker=dict(color='rgb(255, 102, 102)'),  # Kırmızı renk
        )

        layout = go.Layout(
            title=f'Bar Plot for {col}',
            xaxis=dict(title=col),
            yaxis=dict(title='Count'),
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )

        fig = go.Figure(data=[trace], layout=layout)
        plot_html = fig.to_html(full_html=False)
        cat_plots[col] = plot_html

    return cat_plots

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "GET":
        return render_template("upload.html")
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for("analyze", filename=file.filename))



@app.route("/analyze/<filename>")
def analyze(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    data = pd.read_csv(filepath)

    summary = data.describe().to_html()
    cat_columns, num_columns, cat_but_car = grab_col_names(data)

    # Boxplot
    box_plots = []
    for col in num_columns:
        box_plot = go.Box(
            y=data[col],
            name=col,
            marker=dict(color="rgb(255,102,102)")  # kırmızı renk
        )
        box_plots.append(box_plot)
    plot_div_box_plot = go.Figure(data=box_plots).update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    ).to_html(full_html=False)

    histograms, violin_plots = create_histograms_and_violin_plots(data, num_columns)

    plot_div_histograms = []
    for col, histogram in zip(num_columns, histograms):
        fig = go.Figure(data=[histogram])
        fig.update_layout(
            title={"text": f"{col}", "x": 0.5},
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        plot_div_histograms.append(fig.to_html(full_html=False))

    plot_div_violin_plots = []
    for col, violin_plot in zip(num_columns, violin_plots):
        fig = go.Figure(data=[violin_plot])
        fig.update_layout(
            title={"text": f"{col}", "x": 0.5},
            xaxis={"visible": False},
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        plot_div_violin_plots.append(fig.to_html(full_html=False))

    cat_analysis = {col: data[col].value_counts() for col in cat_columns}

    # Kategorik kolonların grafiklerini çiz
    cat_plots = create_categorical_plots(data, cat_columns)

    return render_template("analyze.html", filename=filename, tables=[summary], plot_div_box_plot=plot_div_box_plot,
                           plot_div_histograms=plot_div_histograms, plot_div_violin_plots=plot_div_violin_plots,
                           cat_plots=cat_plots, num_columns=num_columns, cat_columns=cat_columns,
                           cat_analysis=cat_analysis, data=data)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Form verilerini al
        float_features = [float(x) for x in request.form.values()]
        # Değerlerin logaritmasını al
        log_features = [log(x) for x in float_features]
        features = [np.array(log_features)]
        # Tahmini hesapla
        prediction = model.predict(features)
        # Sonucu template'e gönder
        return render_template("predict.html", prediction_text="The predicted value is {:.2f}".format(prediction[0]))
    return render_template("predict.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        # Burada giriş doğrulama işlemi yapın
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot-password.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/financial_analysis", methods=["GET", "POST"])
def financial_analysis():
    if request.method == "POST":
        income = float(request.form.get('income'))
        expense = float(request.form.get('expense'))
        net_profit = income - expense
        return render_template("finansalanaliz.html", result=f"Net profit is {net_profit:.2f}")
    return render_template("finansalanaliz.html")

# @app.route("/predict", methods=["GET", "POST"])
# async def predict():
#     if request.method == "POST":
#         # Form verilerini al
#         form_values = await request.form
#         float_features = [float(x) for x in form_values.values()]
#         # Değerlerin logaritmasını al
#         log_features = [log(x) for x in float_features]
#         features = [np.array(log_features)]
#         # Tahmini hesapla
#         prediction = model.predict(features)
#         # Sonucu template'e gönder
#         return await render_template("predict.html", prediction_text="The predicted value is {:.2f}".format(prediction[0]))
#     return await render_template("predict.html")

# @app.route("/login", methods=["GET", "POST"])
# async def login():
#     if request.method == "POST":
#         form_values = await request.form
#         email = form_values.get('email')
#         password = form_values.get('password')
#         # Burada giriş doğrulama işlemi yapın
#         return redirect(url_for("home"))
#     return await render_template("login.html")

# @app.route("/forgot-password")
# async def forgot_password():
#     return await render_template("forgot-password.html")

# @app.route("/register")
# async def register():
#     return await render_template("register.html")






# @app.route("/financial_analysis", methods=["GET", "POST"])
# async def financial_analysis():
#     if request.method == "POST":
#         form_values = await request.form
#         income = float(form_values.get('income'))
#         expense = float(form_values.get('expense'))
#         net_profit = income - expense
#         return await render_template("finansalanaliz.html", result=f"Net profit is {net_profit:.2f}")
#     return await render_template("finansalanaliz.html")

if __name__ == "__main__":
    app.run(debug=True)

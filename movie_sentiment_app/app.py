from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        review = request.form["review"]
        result = model.predict([review])[0]
        prediction = "Positive 😊" if result == 1 else "Negative 😞"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

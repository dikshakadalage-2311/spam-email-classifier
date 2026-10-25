
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained spam classification model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "spam_model.joblib"
)

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email", "").strip()

        if email_text:
            result = model.predict([email_text])[0]

            if str(result).lower() in ["spam", "1", "junk"]:
                prediction = "Spam Email"
            else:
                prediction = "Not Spam (Ham)"

    return render_template(
        "index.html",
        prediction=prediction,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)

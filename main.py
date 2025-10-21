from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import urllib.request
import os
import gdown
app = Flask(__name__)

# Ensure directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)

MODEL_PATH = "random_forest_log_model.pkl"
MODEL_URL = os.environ.get('MODEL_URL')


def download_model():
    """Download model from Google Drive if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        try:
            # Use gdown for reliable Google Drive downloads
            file_id = "1KatsClmhiVf4uGogExjE7ioW2VMUr4yR"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise Exception("Failed to download model. Please check the file ID.")
    else:
        print("✅ Model already exists locally.")

download_model()

# Load the model
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

def generate_prediction_plot(y_test, y_pred, filepath="static/actual_vs_pred.png"):
    """Generates and saves the Actual vs. Predicted scatter plot."""
    max_val = max(np.max(y_test), np.max(y_pred)) * 1.05

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred,
                color='#1f77b4',
                alpha=0.7,
                s=120,
                edgecolors='w',
                label='Predictions')

    plt.plot([0, max_val], [0, max_val],
             color='#ff7f0e',
             linestyle='--',
             linewidth=2,
             label='Perfect Prediction')

    plt.xlabel("Actual Median Value ($10,000s)", fontsize=12)
    plt.ylabel("Predicted Median Value ($10,000s)", fontsize=12)
    plt.title("Model Performance: Actual vs. Predicted House Prices", fontsize=14, pad=15)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(filepath, dpi=150)
    plt.close()

def check_prediction_reasonable(prediction):
    """Check if prediction is in a reasonable range."""
    if prediction < 0:
        return "⚠️ Prediction is negative! Please check your input values."
    elif prediction > 50:
        return f"⚠️ Prediction unusually high (${prediction * 10000:,.0f})! Please verify your input values."
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    warning_message = None

    if request.method == "POST":
        try:
            features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                        'Population', 'AveOccup', 'Latitude', 'Longitude']

            input_data = [[float(request.form[f]) for f in features]]
            df = pd.DataFrame(input_data, columns=features)

            pred_log = model.predict(df)
            prediction = np.expm1(pred_log)[0]

            warning_message = check_prediction_reasonable(prediction)
            generate_prediction_plot(y_test=[prediction], y_pred=[prediction])

        except Exception as e:
            warning_message = f" Error processing prediction: {str(e)}"
            prediction = None

    return render_template("index.html",
                           prediction=round(prediction, 2) if prediction else None,
                           warning=warning_message)


if __name__ == '__main__':
    app.run(debug=False)

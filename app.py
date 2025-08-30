import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd

from models.ml_service import MLService

# Config
BASE = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
MODEL_FOLDER = os.path.join(BASE, "models_saved")
IMAGE_FOLDER = os.path.join(BASE, "static", "images")
ALLOWED_EXT = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "cambiame_por_una_mas_segura"

ml = MLService(model_dir=MODEL_FOLDER, image_dir=IMAGE_FOLDER)

# keep last dataset metadata to generate dynamic form for single prediction
LAST_DATASET_META = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/")
def home():
    uploads = os.listdir(UPLOAD_FOLDER)
    models = ml.list_models()
    return render_template("home.html", uploads=uploads, models=models)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "warning")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "warning")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                df = pd.read_csv(path)
                preview = df.head(8).to_html(classes="table table-striped", index=False)
                flash(f"Archivo {filename} subido ({df.shape[0]} filas, {df.shape[1]} columnas).", "success")
                return render_template("upload.html", uploaded=True, filename=filename, preview=preview, cols=list(df.columns))
            except Exception as e:
                flash(f"Error leyendo CSV: {e}", "danger")
                return redirect(request.url)
        else:
            flash("Formato no permitido. Usa CSV.", "danger")
            return redirect(request.url)
    return render_template("upload.html", uploaded=False)

@app.route("/train", methods=["GET", "POST"])
def train():
    uploads = os.listdir(UPLOAD_FOLDER)
    if request.method == "POST":
        dataset_file = request.form.get("dataset")
        if not dataset_file:
            flash("Selecciona un dataset.", "warning")
            return redirect(request.url)
        dataset_path = os.path.join(UPLOAD_FOLDER, dataset_file)

        algorithm = request.form.get("algorithm", "id3")
        validation = request.form.get("validation", "holdout")
        k_folds = int(request.form.get("k_folds", 5) or 5)
        knn_k = int(request.form.get("knn_k", 5) or 5)
        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None

        model_name = f"{algorithm}_{uuid.uuid4().hex[:8]}.joblib"
        try:
            result = ml.train_from_csv(
                csv_path=dataset_path,
                target_col=None,  # automatic: last column
                algorithm=algorithm,
                validation=("holdout" if validation == "holdout" else "kfold"),
                k_folds=k_folds,
                knn_k=knn_k,
                max_depth=max_depth,
                model_name=model_name
            )
            # Save last dataset meta so we can render dynamic predict form
            LAST_DATASET_META["feature_names"] = result["feature_names"]
            LAST_DATASET_META["target_col"] = result["target_col"]
            LAST_DATASET_META["model_path"] = result["model_path"]
            flash("Entrenamiento completado.", "success")
            return render_template("results.html", result=result)
        except Exception as e:
            flash(f"Error en entrenamiento: {e}", "danger")
            return redirect(request.url)

    previews = []
    for f in uploads:
        try:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f), nrows=3)
            previews.append((f, list(df.columns)))
        except Exception:
            previews.append((f, []))
    return render_template("train.html", datasets=previews)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    models = ml.list_models()
    single_result = None
    batch_file = None
    batch_preview = None
    # get feature names for dynamic form if available
    feature_names = LAST_DATASET_META.get("feature_names", [])
    if request.method == "POST":
        model_file = request.form.get("model")
        if not model_file:
            flash("Selecciona un modelo", "warning")
            return redirect(request.url)
        model_path = os.path.join(MODEL_FOLDER, model_file)
        mode = request.form.get("mode", "single")
        if mode == "single":
            # gather inputs for each feature name
            data = {}
            for col in feature_names:
                val = request.form.get(f"col__{col}")
                if val is None:
                    val = ""
                data[col] = val
            df_in = pd.DataFrame([data])
            try:
                preds = ml.predict_from_model(model_path, df_in)
                single_result = preds.to_html(index=False)
            except Exception as e:
                flash(f"Error en predicción single: {e}", "danger")
                return redirect(request.url)
        else:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Sube un CSV para predicción batch", "warning")
                return redirect(request.url)
            if not allowed_file(file.filename):
                flash("Solo CSV permitido", "danger")
                return redirect(request.url)
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                df_batch = pd.read_csv(path)
                preds = ml.predict_from_model(os.path.join(MODEL_FOLDER, model_file), df_batch)
                out_name = f"preds_{uuid.uuid4().hex[:6]}.csv"
                out_path = os.path.join(UPLOAD_FOLDER, out_name)
                preds.to_csv(out_path, index=False)
                batch_file = out_name
                batch_preview = preds.head(10).to_html(index=False)
                flash("Predicción batch completada.", "success")
            except Exception as e:
                flash(f"Error predicción batch: {e}", "danger")
                return redirect(request.url)

    return render_template("predict.html", models=models, feature_names=feature_names, single_result=single_result, batch_file=batch_file, batch_preview=batch_preview)

@app.route("/visualize/<model_file>")
def visualize(model_file):
    # mostrará imágenes generadas durante entrenamiento asociadas al modelo
    model_path = os.path.join(MODEL_FOLDER, model_file)
    # construir nombres de imagen por prefijo
    prefix = os.path.splitext(model_file)[0]
    cm = os.path.join(IMAGE_FOLDER, f"cm_{prefix}.png")
    dist = os.path.join(IMAGE_FOLDER, f"dist_{prefix}.png")
    hist = os.path.join(IMAGE_FOLDER, f"hist_{prefix}.png")
    tree = os.path.join(IMAGE_FOLDER, f"tree_{prefix}.png")
    exists = { "cm": os.path.exists(cm), "dist": os.path.exists(dist), "hist": os.path.exists(hist), "tree": os.path.exists(tree) }
    return render_template("visualizations.html", model_file=model_file, cm=cm if exists["cm"] else None,
                           dist=dist if exists["dist"] else None,
                           hist=hist if exists["hist"] else None,
                           tree=tree if exists["tree"] else None)

@app.route("/download/model/<name>")
def download_model(name):
    return send_from_directory(MODEL_FOLDER, name, as_attachment=True)

@app.route("/download/preds/<name>")
def download_preds(name):
    return send_from_directory(UPLOAD_FOLDER, name, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

import os
import uuid
import json
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

# Global storage for model metadata
TRAINED_MODELS = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_model_metadata(model_name, metadata):
    """Save model metadata to persistent storage"""
    TRAINED_MODELS[model_name] = metadata
    # Also save to file for persistence
    metadata_file = os.path.join(MODEL_FOLDER, f"{model_name}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_model_metadata(model_name):
    """Load model metadata from storage"""
    if model_name in TRAINED_MODELS:
        return TRAINED_MODELS[model_name]
    
    # Try to load from file
    metadata_file = os.path.join(MODEL_FOLDER, f"{model_name}.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            TRAINED_MODELS[model_name] = metadata
            return metadata
    return None

def load_all_model_metadata():
    """Load all model metadata on startup"""
    for filename in os.listdir(MODEL_FOLDER):
        if filename.endswith('.json'):
            model_name = filename.replace('.json', '')
            load_model_metadata(model_name)

# Load existing model metadata on startup
load_all_model_metadata()

@app.route("/")
def home():
    uploads = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    models = ml.list_models()
    
    # Add metadata to models for display
    model_info = []
    for model in models:
        metadata = load_model_metadata(model)
        if metadata:
            model_info.append({
                'filename': model,
                'algorithm': metadata.get('algorithm', 'Unknown'),
                'validation': metadata.get('validation_method', 'Unknown'),
                'accuracy': metadata.get('metrics', {}).get('accuracy', 'N/A')
            })
        else:
            model_info.append({
                'filename': model,
                'algorithm': 'Unknown',
                'validation': 'Unknown', 
                'accuracy': 'N/A'
            })
    
    return render_template("home.html", uploads=uploads, models=model_info)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No se seleccionó ningún archivo", "warning")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No se seleccionó ningún archivo", "warning")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                df = pd.read_csv(path)
                preview = df.head(8).to_html(classes="table table-striped", index=False)
                flash(f"Archivo {filename} subido exitosamente ({df.shape[0]} filas, {df.shape[1]} columnas).", "success")
                return render_template("upload.html", uploaded=True, filename=filename, 
                                     preview=preview, cols=list(df.columns), shape=df.shape)
            except Exception as e:
                flash(f"Error leyendo CSV: {e}", "danger")
                return redirect(request.url)
        else:
            flash("Formato no permitido. Solo se acepta CSV.", "danger")
            return redirect(request.url)
    return render_template("upload.html", uploaded=False)

@app.route("/train", methods=["GET", "POST"])
def train():
    uploads = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    
    if request.method == "POST":
        dataset_file = request.form.get("dataset")
        if not dataset_file:
            flash("Debe seleccionar un dataset.", "warning")
            return redirect(request.url)
        
        dataset_path = os.path.join(UPLOAD_FOLDER, dataset_file)
        if not os.path.exists(dataset_path):
            flash("El dataset seleccionado no existe.", "danger")
            return redirect(request.url)

        # Training parameters
        algorithm = request.form.get("algorithm", "id3")
        validation = request.form.get("validation", "holdout")
        k_folds = int(request.form.get("k_folds", 5) or 5)
        knn_k = int(request.form.get("knn_k", 5) or 5)
        max_depth = request.form.get("max_depth")
        max_depth = int(max_depth) if max_depth else None

        # Generate descriptive model name
        algo_name = "ID3" if algorithm == "id3" else "KNN"
        validation_name = "Holdout" if validation == "holdout" else f"{k_folds}Fold"
        model_name = f"{algo_name}_{validation_name}_{uuid.uuid4().hex[:8]}.joblib"
        
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
            
            # Save comprehensive metadata
            metadata = {
                'algorithm': algorithm,
                'validation_method': validation,
                'k_folds': k_folds if validation == 'kfold' else None,
                'knn_k': knn_k if algorithm == 'knn' else None,
                'max_depth': max_depth,
                'feature_names': result["feature_names"],
                'target_col': result["target_col"],
                'dataset_file': dataset_file,
                'metrics': result["metrics"]
            }
            save_model_metadata(model_name, metadata)
            
            flash("Entrenamiento completado exitosamente.", "success")
            return render_template("results.html", result=result, model_name=model_name)
            
        except Exception as e:
            flash(f"Error durante el entrenamiento: {e}", "danger")
            return redirect(request.url)

    # Get dataset previews with column information
    dataset_info = []
    for f in uploads:
        try:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f), nrows=3)
            dataset_info.append({
                'filename': f,
                'columns': list(df.columns),
                'target_col': df.columns[-1],  # Last column assumed as target
                'shape': (len(pd.read_csv(os.path.join(UPLOAD_FOLDER, f))), len(df.columns))
            })
        except Exception:
            dataset_info.append({
                'filename': f,
                'columns': [],
                'target_col': 'Unknown',
                'shape': (0, 0)
            })
            
    return render_template("train.html", datasets=dataset_info)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    models = ml.list_models()
    single_result = None
    batch_file = None
    batch_preview = None
    
    # Get model information for dropdown
    model_options = []
    for model in models:
        metadata = load_model_metadata(model)
        if metadata:
            algo = metadata.get('algorithm', 'Unknown')
            val = metadata.get('validation_method', 'Unknown')
            acc = metadata.get('metrics', {}).get('accuracy', 'N/A')
            if isinstance(acc, float):
                acc = f"{acc:.3f}"
            model_options.append({
                'filename': model,
                'display_name': f"{model} ({algo}, {val}, Acc: {acc})",
                'metadata': metadata
            })
        else:
            model_options.append({
                'filename': model,
                'display_name': model,
                'metadata': None
            })
    
    selected_model_metadata = None
    if request.method == "POST":
        model_file = request.form.get("model")
        if not model_file:
            flash("Debe seleccionar un modelo", "warning")
            return redirect(request.url)
            
        model_path = os.path.join(MODEL_FOLDER, model_file)
        if not os.path.exists(model_path):
            flash("El modelo seleccionado no existe.", "danger")
            return redirect(request.url)
            
        selected_model_metadata = load_model_metadata(model_file)
        mode = request.form.get("mode", "single")
        
        if mode == "single":
            if not selected_model_metadata or 'feature_names' not in selected_model_metadata:
                flash("No se encontraron metadatos del modelo. Vuelva a entrenar el modelo.", "warning")
                return redirect(request.url)
                
            # Gather inputs for each feature
            feature_names = selected_model_metadata['feature_names']
            data = {}
            for col in feature_names:
                val = request.form.get(f"col__{col}")
                if val is None or val.strip() == "":
                    flash(f"Debe completar el campo '{col}'", "warning")
                    return render_template("predict.html", models=model_options, 
                                         selected_model=model_file, 
                                         selected_metadata=selected_model_metadata)
                data[col] = val.strip()
                
            df_in = pd.DataFrame([data])
            try:
                preds = ml.predict_from_model(model_path, df_in)
                single_result = preds.to_html(classes="table table-striped", index=False)
                flash("Clasificación completada.", "success")
            except Exception as e:
                flash(f"Error en la clasificación: {e}", "danger")
                return redirect(request.url)
                
        else:  # batch mode
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Debe subir un archivo CSV para clasificación por lotes", "warning")
                return redirect(request.url)
            if not allowed_file(file.filename):
                flash("Solo se permiten archivos CSV", "danger")
                return redirect(request.url)
                
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            
            try:
                df_batch = pd.read_csv(path)
                preds = ml.predict_from_model(model_path, df_batch)
                out_name = f"predictions_{uuid.uuid4().hex[:6]}.csv"
                out_path = os.path.join(UPLOAD_FOLDER, out_name)
                preds.to_csv(out_path, index=False)
                batch_file = out_name
                batch_preview = preds.head(10).to_html(classes="table table-striped", index=False)
                flash("Clasificación por lotes completada.", "success")
            except Exception as e:
                flash(f"Error en clasificación por lotes: {e}", "danger")
                return redirect(request.url)

    return render_template("predict.html", 
                         models=model_options,
                         selected_model=request.form.get("model") if request.method == "POST" else None,
                         selected_metadata=selected_model_metadata,
                         single_result=single_result, 
                         batch_file=batch_file, 
                         batch_preview=batch_preview)

@app.route("/model_details/<model_name>")
def model_details(model_name):
    """Show detailed information about a specific model"""
    metadata = load_model_metadata(model_name)
    if not metadata:
        flash("No se encontraron metadatos para este modelo.", "warning")
        return redirect(url_for('home'))
    
    return render_template("model_details.html", model_name=model_name, metadata=metadata)

@app.route("/visualize/<model_file>")
def visualize(model_file):
    """Show visualizations for a model"""
    model_path = os.path.join(MODEL_FOLDER, model_file)
    if not os.path.exists(model_path):
        flash("El modelo no existe.", "danger")
        return redirect(url_for('home'))
    
    # Build image paths
    prefix = os.path.splitext(model_file)[0]
    images = {
        'cm': os.path.join(IMAGE_FOLDER, f"cm_{prefix}.png"),
        'dist': os.path.join(IMAGE_FOLDER, f"dist_{prefix}.png"), 
        'hist': os.path.join(IMAGE_FOLDER, f"hist_{prefix}.png"),
        'tree': os.path.join(IMAGE_FOLDER, f"tree_{prefix}.png")
    }
    
    # Check which images exist
    available_images = {}
    for key, path in images.items():
        if os.path.exists(path):
            available_images[key] = f"static/images/{os.path.basename(path)}"
    
    metadata = load_model_metadata(model_file)
    return render_template("visualizations.html", 
                         model_file=model_file,
                         images=available_images,
                         metadata=metadata)

@app.route("/download/model/<name>")
def download_model(name):
    return send_from_directory(MODEL_FOLDER, name, as_attachment=True)

@app.route("/download/preds/<name>")
def download_preds(name):
    return send_from_directory(UPLOAD_FOLDER, name, as_attachment=True)

@app.route("/delete_model/<model_name>", methods=["POST"])
def delete_model(model_name):
    """Delete a model and its associated files"""
    try:
        # Delete model file
        model_path = os.path.join(MODEL_FOLDER, model_name)
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Delete metadata file
        metadata_path = os.path.join(MODEL_FOLDER, f"{model_name}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Delete associated images
        prefix = os.path.splitext(model_name)[0]
        for img_type in ['cm', 'dist', 'hist', 'tree']:
            img_path = os.path.join(IMAGE_FOLDER, f"{img_type}_{prefix}.png")
            if os.path.exists(img_path):
                os.remove(img_path)
        
        # Remove from memory
        if model_name in TRAINED_MODELS:
            del TRAINED_MODELS[model_name]
        
        flash(f"Modelo {model_name} eliminado exitosamente.", "success")
    except Exception as e:
        flash(f"Error eliminando modelo: {e}", "danger")
    
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
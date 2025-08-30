import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set(style="whitegrid")

class MLService:
    def __init__(self, model_dir="models_saved", image_dir="static/images"):
        self.model_dir = model_dir
        self.image_dir = image_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def _prepare_preprocessor(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", num_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", cat_pipeline, categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor, numeric_cols, categorical_cols

    def _make_pipeline(self, X: pd.DataFrame, algorithm="id3", knn_k=5, max_depth=None):
        preprocessor, _, _ = self._prepare_preprocessor(X)
        if algorithm.lower() in ["id3", "decisiontree", "dt"]:
            clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=max_depth)
        elif algorithm.lower() in ["knn", "k-nn"]:
            clf = KNeighborsClassifier(n_neighbors=knn_k)
        else:
            raise ValueError("Algoritmo no soportado")
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        return pipe

    def _get_feature_names(self, preprocessor: ColumnTransformer, X: pd.DataFrame):
        # intenta construir nombres de features despues de ColumnTransformer
        try:
            # sklearn >=1.0: get_feature_names_out
            names = preprocessor.get_feature_names_out()
            return list(names)
        except Exception:
            # fallback: manual
            numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            names = numeric_cols.copy()
            # for each categorical, approximate onehot names
            if categorical_cols:
                # if onehot encoder present, try to extract categories
                for name, transformer, cols in preprocessor.transformers_:
                    if name == "cat":
                        try:
                            ohe = transformer.named_steps["onehot"]
                            cats = ohe.categories_
                            for col, catvals in zip(cols, cats):
                                for v in catvals:
                                    names.append(f"{col}__{v}")
                        except Exception:
                            for col in cols:
                                names.append(col)
            return names

    def train_from_csv(self, csv_path, target_col=None, algorithm="id3", validation="holdout", k_folds=5, knn_k=5, max_depth=None, model_name=None):
        df = pd.read_csv(csv_path)
        if target_col is None:
            # elegir automaticamente: si hay columna categórica al final -> target,
            # si ultima es numérica -> target, por defecto la ultima
            target_col = df.columns[-1]

        if target_col not in df.columns:
            raise ValueError("Target column no encontrada en dataset")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Pipeline
        pipe = self._make_pipeline(X, algorithm=algorithm, knn_k=knn_k, max_depth=max_depth)

        metrics = {}
        cm = None

        if validation == "holdout":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y if y.nunique() > 1 else None, random_state=42)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            metrics["accuracy"] = float(acc)
            metrics["report"] = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
        else:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            metrics["cv_scores"] = [float(s) for s in scores]
            metrics["cv_mean"] = float(np.mean(scores))
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            metrics["report"] = classification_report(y, y_pred, output_dict=True)
            cm = confusion_matrix(y, y_pred)
            pipe.fit(X, y)

        # Guardar modelo
        if model_name is None:
            model_name = f"model_{algorithm}_{int(np.random.randint(1e6))}.joblib"
        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(pipe, model_path)

        # Guardar confusion matrix
        img_name = f"cm_{os.path.splitext(model_name)[0]}.png"
        img_path = os.path.join(self.image_dir, img_name)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Matriz de Confusión")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        # Graficos de datos: distribucion de clases y histogramas
        # 1) distribucion de clases
        dist_name = f"dist_{os.path.splitext(model_name)[0]}.png"
        dist_path = os.path.join(self.image_dir, dist_name)
        plt.figure(figsize=(6,4))
        sns.countplot(x=y)
        plt.title("Distribución de clases (target)")
        plt.tight_layout()
        plt.savefig(dist_path)
        plt.close()

        # 2) histogramas (numeric)
        hist_name = f"hist_{os.path.splitext(model_name)[0]}.png"
        hist_path = os.path.join(self.image_dir, hist_name)
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) > 0:
            plt.figure(figsize=(8, 4 + 1*len(numeric_cols)))
            X[numeric_cols].hist(bins=15, figsize=(8, max(4, 2*len(numeric_cols))))
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
        else:
            hist_path = None

        # Árbol de decisión (si aplica)
        tree_path = None
        if algorithm.lower() in ["id3", "decisiontree", "dt"]:
            try:
                # intentamos obtener nombres de features tras el preprocesado
                feature_names = self._get_feature_names(pipe.named_steps["pre"], X)
                tree_name = f"tree_{os.path.splitext(model_name)[0]}.png"
                tree_path = os.path.join(self.image_dir, tree_name)
                plt.figure(figsize=(14,8))
                plot_tree(pipe.named_steps["clf"], feature_names=feature_names, class_names=[str(c) for c in np.unique(y)], filled=True, rounded=True, fontsize=8)
                plt.tight_layout()
                plt.savefig(tree_path)
                plt.close()
            except Exception:
                tree_path = None

        return {
            "model_path": model_path,
            "confusion_image": img_path,
            "dist_image": dist_path,
            "hist_image": hist_path,
            "tree_image": tree_path,
            "metrics": metrics,
            "target_col": target_col,
            "feature_names": list(X.columns)
        }

    def list_models(self):
        files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
        return files

    def load_model(self, model_path):
        return joblib.load(model_path)

    def predict_from_model(self, model_path, X_df: pd.DataFrame):
        pipe = self.load_model(model_path)
        # ensure dataframe columns present: pass through pipeline (it handles unknown categories)
        preds = pipe.predict(X_df)
        out = X_df.copy()
        out["prediction"] = preds
        return out

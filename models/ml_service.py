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

sns.set_style("whitegrid")

class MLService:
    def __init__(self, model_dir="models_saved", image_dir="static/images"):
        self.model_dir = model_dir
        self.image_dir = image_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def _prepare_preprocessor(self, X: pd.DataFrame):
        """Prepara el preprocesador para datos numéricos y categóricos"""
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        transformers = []
        
        # Pipeline para variables numéricas
        if numeric_cols:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipeline, numeric_cols))
        
        # Pipeline para variables categóricas
        if categorical_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

        if not transformers:
            raise ValueError("No se encontraron columnas válidas para procesar")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor, numeric_cols, categorical_cols

    def _make_pipeline(self, X: pd.DataFrame, algorithm="id3", knn_k=5, max_depth=None):
        """Crea el pipeline completo con preprocesamiento y clasificador"""
        preprocessor, _, _ = self._prepare_preprocessor(X)
        
        # Selección del algoritmo
        if algorithm.lower() in ["id3", "decisiontree", "dt"]:
            clf = DecisionTreeClassifier(
                criterion="entropy", 
                random_state=42, 
                max_depth=max_depth
            )
        elif algorithm.lower() in ["knn", "k-nn"]:
            clf = KNeighborsClassifier(n_neighbors=knn_k)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")
        
        pipe = Pipeline([
            ("preprocessor", preprocessor), 
            ("classifier", clf)
        ])
        return pipe

    def _get_feature_names(self, preprocessor: ColumnTransformer, X: pd.DataFrame):
        """Intenta construir nombres de features después de ColumnTransformer"""
        try:
            # sklearn >=1.2: get_feature_names_out
            if hasattr(preprocessor, 'get_feature_names_out'):
                names = preprocessor.get_feature_names_out()
                return list(names)
        except Exception:
            pass
        
        # Fallback manual
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        names = numeric_cols.copy()
        
        # Para columnas categóricas, aproximar nombres de onehot
        for name, transformer, cols in preprocessor.transformers_:
            if name == "cat" and hasattr(transformer, 'named_steps'):
                try:
                    ohe = transformer.named_steps["onehot"]
                    if hasattr(ohe, 'categories_'):
                        for col, categories in zip(cols, ohe.categories_):
                            for category in categories:
                                names.append(f"{col}_{category}")
                except Exception:
                    # Si falla, usar nombres de columnas originales
                    for col in cols:
                        names.append(col)
        
        return names

    def train_from_csv(self, csv_path, target_col=None, algorithm="id3", validation="holdout", 
                      k_folds=5, knn_k=5, max_depth=None, model_name=None):
        """
        Entrena un modelo desde un archivo CSV
        
        Args:
            csv_path: Ruta del archivo CSV
            target_col: Nombre de la columna objetivo (si None, usa la última columna)
            algorithm: 'id3' o 'knn'
            validation: 'holdout' o 'kfold'
            k_folds: Número de folds para validación cruzada
            knn_k: Número de vecinos para KNN
            max_depth: Profundidad máxima para árbol de decisión
            model_name: Nombre del archivo del modelo
        """
        try:
            # Cargar dataset
            df = pd.read_csv(csv_path)
            print(f"Dataset cargado: {df.shape}")
            
            # Determinar columna objetivo
            if target_col is None:
                target_col = df.columns[-1]
            
            if target_col not in df.columns:
                raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset")
            
            # Separar features y target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            print(f"Features: {list(X.columns)}")
            print(f"Target: {target_col}")
            print(f"Clases únicas: {y.unique()}")
            
            # Verificar que hay suficientes datos
            if len(df) < 5:
                raise ValueError("El dataset debe tener al menos 5 filas")
            
            # Crear pipeline
            pipe = self._make_pipeline(X, algorithm=algorithm, knn_k=knn_k, max_depth=max_depth)
            
            # Inicializar métricas y matriz de confusión
            metrics = {}
            cm = None
            
            # Aplicar validación seleccionada
            if validation == "holdout":
                # Hold-out validation
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, 
                        stratify=y if len(y.unique()) > 1 else None, 
                        random_state=42
                    )
                except ValueError:
                    # Si stratify falla, usar split sin estratificar
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )
                
                # Entrenar y predecir
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                
                # Calcular métricas
                acc = accuracy_score(y_test, y_pred)
                metrics["accuracy"] = float(acc)
                metrics["validation_method"] = "holdout"
                
                try:
                    metrics["report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
                except Exception as e:
                    print(f"Warning: Error en classification_report: {e}")
                    metrics["report"] = {}
                    cm = confusion_matrix(y_test, y_pred)
                    
            else:
                # K-Fold Cross Validation
                try:
                    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
                    metrics["cv_scores"] = [float(s) for s in scores]
                    metrics["cv_mean"] = float(np.mean(scores))
                    metrics["cv_std"] = float(np.std(scores))
                    metrics["accuracy"] = float(np.mean(scores))  # Para consistencia
                    metrics["validation_method"] = f"{k_folds}-fold"
                    
                    # Predicciones para matriz de confusión
                    y_pred = cross_val_predict(pipe, X, y, cv=cv)
                    metrics["report"] = classification_report(y, y_pred, output_dict=True, zero_division=0)
                    cm = confusion_matrix(y, y_pred)
                    
                except ValueError as e:
                    # Fallback a validación simple si K-fold falla
                    print(f"Warning: K-fold falló, usando holdout: {e}")
                    return self.train_from_csv(csv_path, target_col, algorithm, "holdout", 
                                             k_folds, knn_k, max_depth, model_name)
                
                # Entrenar en todo el dataset
                pipe.fit(X, y)
            
            # Generar nombre del modelo si no se proporciona
            if model_name is None:
                algo_name = algorithm.upper()
                val_name = "holdout" if validation == "holdout" else f"{k_folds}fold"
                model_name = f"{algo_name}_{val_name}_{np.random.randint(1000, 9999)}.joblib"
            
            # Guardar modelo
            model_path = os.path.join(self.model_dir, model_name)
            joblib.dump(pipe, model_path)
            print(f"Modelo guardado en: {model_path}")
            
            # Generar visualizaciones
            prefix = os.path.splitext(model_name)[0]
            image_paths = self._generate_visualizations(X, y, cm, prefix)
            
            # Generar visualización del árbol si es ID3
            tree_image = None
            if algorithm.lower() in ["id3", "decisiontree", "dt"]:
                tree_image = self._generate_tree_visualization(pipe, X, y, prefix)
            
            # Retornar resultados
            result = {
                "model_path": model_path,
                "confusion_image": image_paths.get("confusion"),
                "dist_image": image_paths.get("distribution"),
                "hist_image": image_paths.get("histogram"),
                "tree_image": tree_image,
                "metrics": metrics,
                "target_col": target_col,
                "feature_names": list(X.columns),
                "algorithm": algorithm,
                "validation_method": validation
            }
            
            print("Entrenamiento completado exitosamente")
            return result
            
        except Exception as e:
            print(f"Error en train_from_csv: {e}")
            raise

    def _generate_visualizations(self, X, y, cm, prefix):
        """Genera visualizaciones del modelo y datos"""
        image_paths = {}
        
        try:
            # 1. Matriz de confusión
            if cm is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                           xticklabels=np.unique(y), yticklabels=np.unique(y))
                plt.xlabel("Predicción")
                plt.ylabel("Valor Real")
                plt.title("Matriz de Confusión")
                plt.tight_layout()
                cm_path = os.path.join(self.image_dir, f"cm_{prefix}.png")
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                image_paths["confusion"] = cm_path
        except Exception as e:
            print(f"Error generando matriz de confusión: {e}")
        
        try:
            # 2. Distribución de clases
            plt.figure(figsize=(8, 6))
            value_counts = y.value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.xlabel("Clase")
            plt.ylabel("Frecuencia")
            plt.title("Distribución de Clases")
            plt.xticks(rotation=45)
            plt.tight_layout()
            dist_path = os.path.join(self.image_dir, f"dist_{prefix}.png")
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            image_paths["distribution"] = dist_path
        except Exception as e:
            print(f"Error generando distribución: {e}")
        
        try:
            # 3. Histogramas de variables numéricas
            numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for i, col in enumerate(numeric_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                    X[col].hist(bins=20, ax=ax)
                    ax.set_title(f"Distribución de {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frecuencia")
                
                # Ocultar subplots vacíos
                for i in range(len(numeric_cols), n_rows * n_cols):
                    row = i // n_cols
                    col_idx = i % n_cols
                    ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                    ax.set_visible(False)
                
                plt.tight_layout()
                hist_path = os.path.join(self.image_dir, f"hist_{prefix}.png")
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
                image_paths["histogram"] = hist_path
        except Exception as e:
            print(f"Error generando histogramas: {e}")
        
        return image_paths

    def _generate_tree_visualization(self, pipe, X, y, prefix):
        """Genera visualización del árbol de decisión"""
        try:
            # Obtener el clasificador del pipeline
            classifier = pipe.named_steps["classifier"]
            
            if hasattr(classifier, 'tree_'):
                # Intentar obtener nombres de features
                try:
                    feature_names = self._get_feature_names(pipe.named_steps["preprocessor"], X)
                except:
                    feature_names = [f"feature_{i}" for i in range(classifier.n_features_in_)]
                
                # Crear visualización del árbol
                plt.figure(figsize=(20, 12))
                plot_tree(classifier, 
                         feature_names=feature_names[:classifier.n_features_in_], 
                         class_names=[str(c) for c in np.unique(y)], 
                         filled=True, 
                         rounded=True, 
                         fontsize=10)
                plt.title("Árbol de Decisión")
                plt.tight_layout()
                
                tree_path = os.path.join(self.image_dir, f"tree_{prefix}.png")
                plt.savefig(tree_path, dpi=300, bbox_inches='tight')
                plt.close()
                return tree_path
        except Exception as e:
            print(f"Error generando árbol: {e}")
        
        return None

    def list_models(self):
        """Lista todos los modelos guardados"""
        try:
            files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
            return sorted(files)
        except FileNotFoundError:
            return []

    def load_model(self, model_path):
        """Carga un modelo desde archivo"""
        try:
            return joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Error cargando modelo: {e}")

    def predict_from_model(self, model_path, X_df: pd.DataFrame):
        """Realiza predicciones usando un modelo guardado"""
        try:
            # Cargar modelo
            pipe = self.load_model(model_path)
            
            # Realizar predicciones
            predictions = pipe.predict(X_df)
            
            # Crear DataFrame resultado
            result = X_df.copy()
            result["prediction"] = predictions
            
            # Intentar obtener probabilidades si están disponibles
            if hasattr(pipe.named_steps["classifier"], "predict_proba"):
                try:
                    probabilities = pipe.predict_proba(X_df)
                    classes = pipe.named_steps["classifier"].classes_
                    
                    # Agregar columnas de probabilidad
                    for i, cls in enumerate(classes):
                        result[f"prob_{cls}"] = probabilities[:, i]
                        
                    # Agregar confianza (probabilidad máxima)
                    result["confidence"] = np.max(probabilities, axis=1)
                except Exception as e:
                    print(f"Warning: No se pudieron calcular probabilidades: {e}")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error en predicción: {e}")
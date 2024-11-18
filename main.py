import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report 
from sklearn.metrics import  accuracy_score
from sklearn.metrics import  roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # Clean data
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Evaluate model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")


def train_with_mlflow():
    # Cargar la configuración desde config.yml
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Configuración de MLflow
    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run() as run:
        # Cargar datos (supongo que tienes alguna clase para la ingestión de datos)
        ingestion = Ingestion()  # Asegúrate de que tienes esta clase definida
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Limpiar datos
        cleaner = Cleaner()  # Asegúrate de que tienes esta clase definida
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Separar características y etiquetas
        trainer = Trainer()  # Asegúrate de que tienes esta clase definida
        X_train, y_train = trainer.feature_target_separator(train_data)

        # Crear pipeline de modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Paso de escalado
            ('clf', LogisticRegression())  # Modelo base
        ])

        # Extraer los hiperparámetros desde el archivo config.yml
        model_params = config['model']['params']

        # Configuración de GridSearchCV para optimizar los hiperparámetros
        param_grid = {
            'clf__C': model_params.get('C', [0.1, 1.0, 10.0]),  # Hiperparámetro C
            'clf__solver': model_params.get('solver', ['lbfgs', 'liblinear']),  # Solver
            'clf__max_iter': model_params.get('max_iter', [100, 200, 300]),  # Número máximo de iteraciones
        }

        # Usar GridSearchCV para optimizar los hiperparámetros
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Entrenar el modelo con optimización de hiperparámetros
        grid_search.fit(X_train, y_train)

        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_

        # Loguear los mejores parámetros y el mejor score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)

        # Guardar el mejor modelo en MLflow
        mlflow.sklearn.log_model(best_model, "best_model")

        # Evaluar el modelo
        predictor = Predictor()  # Asegúrate de que tienes esta clase definida
        X_test, y_test = predictor.feature_target_separator(test_data)
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Loguear las métricas
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])

        # Etiquetas (tags) de MLflow
        mlflow.set_tag('Model developer', 'prsdm')
        mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')

        # Registrar el modelo en MLflow
        model_name = "churn_model"
        model_uri = f"runs:/{run.info.run_id}/best_model"
        mlflow.register_model(model_uri, model_name)

        logging.info("MLflow tracking completed successfully")

        # Imprimir resultados de la evaluación
        print("\n============= Model Evaluation Results ==============")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc:.4f}")
        print(f"Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Recall: {report['weighted avg']['recall']:.4f}")
        print("=====================================================\n")
        
        print("=====================================================\n")
        print("===¡¡¡LO LOGRE PROFE!!!=====\n")
        print("=====================================================\n")
        
if __name__ == "__main__":
    #main()
    train_with_mlflow()

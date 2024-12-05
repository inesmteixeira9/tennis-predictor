# Usage: python -m pipeline.models.logistic_regression
from libs.data_utils import read_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from .. import PARAMS
import logging

class LogisticRegressionTrainer:
    def __init__(self):
        # Get paths
        self.processed_data_path = PARAMS.data_path.processed.root_dir
        X_train_data_path = self.processed_data_path + PARAMS.data_path.processed.X_train
        y_train_data_path = self.processed_data_path + PARAMS.data_path.processed.y_train
        X_val_data_path = self.processed_data_path + PARAMS.data_path.processed.X_val
        y_val_data_path = self.processed_data_path + PARAMS.data_path.processed.y_val

        # Read data
        self.X_train = read_data(X_train_data_path)
        self.y_train = read_data(y_train_data_path)
        self.X_val = read_data(X_val_data_path)
        self.y_val = read_data(y_val_data_path)

        # Get target
        self.y_train = self.y_train['winner_is_p1'].values
        self.y_train = self.y_train.ravel()

        self.y_val = self.y_val['winner_is_p1'].values
        self.y_val = self.y_val.ravel()

        self.model_params = PARAMS.logistic_regression
        self.model_path = PARAMS.logistic_regression.model_path

    def create_model(self):

        logging.info(f'Creating model with params:\n' + '\n'.join([f'{key}: {value}' for key, value in self.model_params.items()]))
        return LogisticRegression(penalty = self.model_params.penalty,
                                    dual = self.model_params.dual, 
                                    tol = self.model_params.tol,
                                    C = self.model_params.C, 
                                    fit_intercept = self.model_params.fit_intercept, 
                                    intercept_scaling = self.model_params.intercept_scaling, 
                                    class_weight = self.model_params.class_weight, 
                                    random_state = self.model_params.random_state, 
                                    solver = self.model_params.solver,
                                    max_iter = self.model_params.max_iter, 
                                    multi_class = self.model_params.multi_class, 
                                    verbose = self.model_params.verbose, 
                                    warm_start = self.model_params.warm_start, 
                                    n_jobs = self.model_params.n_jobs, 
                                    l1_ratio = self.model_params.l1_ratio
        )
    
    def train_model(self, model):
        return model.fit(self.X_train, self.y_train)
    
    def evaluate(self, trained_model):
        """
        Evaluate the model and return accuracy.
        
        Parameters:
        - trained_model
        """
        train_accuracy = trained_model.score(self.X_train, self.y_train)
        val_accuracy = trained_model.score(self.X_val, self.y_val)

        logging.info(f"Training accuracy: {train_accuracy}")
        logging.info(f"Validation accuracy: {val_accuracy}")


    def save_model(self, trained_model):
        """
        Save the trained model to disk.
        """
        joblib.dump(trained_model, self.model_path)
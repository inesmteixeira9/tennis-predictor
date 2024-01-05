"""


python src/models/train_model.py C:/Users/inesm/projectos/tennis-predictor/data/processed/features_classified.pkl random-forest
"""
import click
import logging
import sys
sys.path.append('src/data/')
from make_dataset import load_data
from make_dataset import save_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import yaml 
import numpy as np
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

config_file_path = "C:/Users/inesm/projectos/tennis-predictor/conf/conf.yaml"
input_filepath = "C:/Users/inesm/projectos/tennis-predictor/data/processed/"
output_filepath = "C:/Users/inesm/projectos/tennis-predictor/models/"
        
with open(config_file_path, "r") as config_file:
    conf = yaml.safe_load(config_file)

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('model', type=click.STRING)
def main(file_path, model):
    """ Build and train classifier model.
        arg: dataset with target and features from (../processed) 
             training configurations
             hyperparameters grid (param_grid)
        return:
             best hyperparameters (best_params)
    """
    
    # Load data using the load_data function
    logger.info('loading features')
    dataset = load_data(input_filepath, file_name=file_path)
    
    # Define training datasets
    y = dataset["winner_is_p1"]    
    
    if file_path == input_filepath + 'features_classified.pkl':
        print(input_filepath + 'features_classified.pkl')
        features_names = conf['classified_features']
        features_names.extend(['match_id'])
        X = dataset[features_names]
    else:
        features_names = conf['features']
        features_names.extend(['match_id'])
        X = dataset[features_names]
        
    X_train_id, X_test_id, y_train, y_test = split_data(X, y)
    X_train = X_train_id.drop(['match_id'], axis = 1)
    X_test = X_test_id.drop(['match_id'], axis = 1)

    # load auxiliar data to use in the custom profit function
    aux_dataset = load_data(input_filepath, file_name='features.pkl')
    # add oddP1 and OddP2 to dataset for custom_profit calculations
    X_train_aux = pd.merge(X_train_id, aux_dataset, on='match_id') 
    X_test_aux = pd.merge(X_test_id, aux_dataset, on='match_id') 

    if model == 'decision-tree':
            # Create a Random Forest classifier
        classifier = DecisionTreeClassifier(class_weight=conf['training']['class_weights'],
                                            random_state=42)
    elif model == 'random-forest':
        # Create a Random Forest classifier
        classifier = RandomForestClassifier(class_weight=conf['training']['class_weights'],
                                            random_state=42)
    
    trained_classifier, best_params = training(X_train, y_train, X_train_aux['OddP1_y'], X_train_aux['OddP2'], classifier, model)

    print(best_params)
    
    # Train the model on the training data
    trained_classifier.fit(X_train, y_train)
    
    profit = custom_profit(y_test, trained_classifier.predict(X_test), X_test_aux['OddP1_y'], X_test_aux['OddP2'])

    if profit > conf["results"]["profit"]:
        # Save trained_classifier
        try:
            joblib.dump(trained_classifier, 'models/random_forest_model.joblib')
        except:
            print('cannot save with joblib')
        try:
            with open('models/random_forest_model.pkl', 'wb') as model_file:
                pickle.dump(trained_classifier, model_file)
        except:
            print('cannot save with pickle')
        print('trained classifier saved in /models')

        # Update the value in conf["results"]["profit"]
        conf["results"]["profit"] = profit  

        # Write the updated configuration back to the file
        with open(config_file_path, "w") as config_file:
            yaml.dump(conf, config_file)

def split_data(X_id, y):
    # Define size of validation data
    train_size = float(conf["training"]["train_size"])
    test_size = conf["training"]["test_size"]
    cv = conf["training"]["cv"]

    if cv != None: 
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_id, y, test_size=test_size, random_state=42)
        logger.info(f"Splitting data randomly into a training ({train_size * 100}%) and a testing ({test_size * 100}%) datasets.")
        return X_train, X_test, y_train, y_test
    else:
        # define validation size
        temp_size = 1 - train_size
        test_val_ratio = test_size / temp_size
        
        # Split data into training, validation and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_id, y, test_size=temp_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_val_ratio, random_state=42)
        logger.info(f"Splitting data randomly into a training ({train_size * 100}%), a validation ({round(1 - train_size - test_size, 2) * 100}%) and a testing ({test_size * 100}%) datasets.")
        return X_train, X_val, X_test, y_train, y_val, y_test


def custom_profit(y, y_pred, Odd_P1, Odd_P2):
    ''' Takes two np.arrays with real and predicted values
    and it's odds and returns the profit. '''

    # Define costs and benefits
    Benefit_TP = Odd_P1 - 1  # Benefit for true positives
    Benefit_TN = Odd_P2 - 1  # Benefit for true negatives
    Cost_FP = -1  # Cost for false positives
    Cost_FN = -1  # Cost for false negatives

    # Calculate TP, TN, FP, FN based on true labels and predictions
    TP = (y == 1) & (y_pred == 1)
    TN = (y == 0) & (y_pred == 0)
    FP = (y == 0) & (y_pred == 1)
    FN = (y == 1) & (y_pred == 0)
    
    profit = Benefit_TP * (TP == True) + Benefit_TN * (TN == True) + Cost_FP * (FP == True) + Cost_FN * (FN == True)
    profitability = 100 * (((profit.sum() + len(y)) / len(y)) - 1)
        
    print('Financial results betting 1€ on each bet')
    print('Invested amount: {:.2f}€'.format(len(y)))
    print('Profit: {:.2f}€'.format(profit.sum()))
    print('profitability: {:.2f}%'.format(profitability))
    return profit.sum()

def training(X, y, OddP1, OddP2, classifier, model):

    # Grid search for hyperparameter tuning
    param_grid = conf[model]['param_grid']

    # Define the objective function
    custom_scorer = lambda estimator, X, y: custom_profit(y, 
                                                         estimator.predict(X), 
                                                         OddP1, 
                                                         OddP2)

    grid_search = GridSearchCV(classifier, param_grid, cv=conf['training']['cv'], scoring=custom_scorer)

    grid_search.fit(X, y)
    
    # Print the best hyperparameters
    print('Best Hyperparameters:', grid_search.best_params_)

    return classifier, grid_search.best_params_





def permutation_importance():
    
    # Assuming trained_model is your KerasRegressor
    result = permutation_importance(trained_model, X_test_scaled, y_test, n_repeats=10, random_state=42)

    # Get feature importances
    importances = result.importances_mean

    feature_names = X_train.columns

    # Plot the feature importances
    plt.figure(figsize=(17, 6))
    plt.bar(range(len(feature_names)), importances, tick_label=feature_names)

    plt.xticks(range(X_test.shape[1]), feature_names, rotation=55)
    plt.xlabel("Feature")
    plt.ylabel("Permutation Importance")
    plt.title("Permutation Importance for each Feature")
    plt.tight_layout()
    plt.savefig('permutation_importances.png')
    plt.show()

def write_report(model):

    # Save the best hyperparameters in an HTML report
    report_path = 'hyperparameter_report.html'
    today_date = datetime.now().strftime('%Y-%m-%d')

    # HTML content with hyperparameter details
    # Save configs tested and results in an HTML report
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <body>
        <h3>Model - {model}</h3>
        <p>Tested parameters: {conf[model]['param_grid']}</p>   
        <p>Best Hyperparameters: {best_params}</p>
        <img src="../reports/permutation-importances.png" alt="Permutation Importances">
        <h3>Profit: {profit.sum():.2f}</h3>
    </body>
    </html>
    '''

    # Write the HTML content to the file
    with open(report_path, 'w') as report_file:
        report_file.write(html_content)

    print(f"Training report saved to: {report_path}")
    


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            # Calculate distances between x_test and all examples in the training set
            distances = np.linalg.norm(self.X_train - x_test, axis=1)

            # Get indices of k-nearest training data points
            k_nearest_indices = np.argsort(distances)[:self.k]

            # Get the labels of the k-nearest training data points
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Predict the class that appears most frequently among the k-nearest neighbors
            predicted_class = np.argmax(np.bincount(k_nearest_labels))
            predictions.append(predicted_class)

        return np.array(predictions)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    main()

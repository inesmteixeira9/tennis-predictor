"""


python src/models/train_model.py C:/Users/inesm/projectos/tennis-predictor/data/processed/ C:/Users/inesm/projectos/tennis-predictor/models/
"""
import click
import logging
import sys
sys.path.append('src/data/')
from make_dataset import load_data
from make_dataset import save_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import yaml 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
from datetime import datetime
import pickle
from pathlib import Path

with open("C:/Users/inesm/projectos/tennis-predictor/conf/conf.yaml", "r") as config_file:
    conf = yaml.safe_load(config_file)

input_filepath = "C:/Users/inesm/projectos/tennis-predictor/data/processed/"
output_filepath = "C:/Users/inesm/projectos/tennis-predictor/models/"

@click.command()
@click.argument('file_name', type=click.Path(exists=True))
def main(input_filepath, output_filepath, dataset):
    """ Build classifier model with optimal hyperparameters for dataset from (../processed) 
        and save trained model in ../models.
    """
    logger = logging.getLogger(__name__)
    
    # Load data using the load_data function
    logger.info('loading features')
    dataset = load_data(input_filepath, file_name=dataset)
    
    # load auxiliar data to use in the custom profit function
    aux_dataset = load_data(processed_filepath, file_name='features.pkl')
    
    # Define training datasets
    y = dataset["winner_is_p1"]    
    features_names = conf['classified_features']
    X = dataset[features_names]
    
    # Create a Random Forest classifier
    classifier = RandomForestClassifier(
        class_weight=conf['training']['class_weights'], # assign a higher weight to the least frequent class (0: P2 wins) to make the model pay more attention to it during training
        random_state=42)
    
    classifier, best_params = training(X_train, y_train, X_train_aux['OddP1_y'], X_train_aux['OddP2_y'])


    final_X_train_id = X_train_id[features_names]
    final_X_train = final_X_train_id.drop(['match_id'], axis = 1)

    final_X_test_id = X_test_id[features_names]
    final_X_test = final_X_test_id.drop(['match_id'], axis = 1)
    
    train_data, val_data, test_data, y_train, y_val, y_test = split_data(features, y)
    
    # Drop col 'match_id' (only needed for evaluation)
    X_train = drop_id(train_data)
    X_val = drop_id(val_data)
    X_test = drop_id(test_data)

    
    decision_tree, best_params = DT_training(X_train, y_train)
    decision_tree, accuracy, report, y_pred, y_test = DT_test(X_train, y_train, X_test, y_test, best_params)
    
    results = pd.DataFrame()
    results['y_pred'] = y_pred
    results['y_test'] = y_test
    print(type(X_test_scaled))
    #results = results.merge(X_test_scaled, left_index=True, right_index=True)
    results = results.merge(test_data, left_index=True, right_index=True)
    print(results)
    #logger.info('saving features')
    #save_data(results, 'results.csv', output_filepath)
        


"""
Save trained models

# Serialize the model
with open('model.pkl', 'wb') as file:
    pickle.dump(trained_model, file)

# Deserialize the model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

"""


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

def training(X, y, OddP1, OddP2):

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



def DT_training(X_train, y_train):
    
    # Create a decision tree classifier
    decision_tree = DecisionTreeClassifier(
        class_weight=conf["decision-tree"]["class_weights"], # assign a higher weight to the least frequent class (0: P2 wins) to make the model pay more attention to it during training
        random_state=42)
    
    # Grid search for hyperparameter tuning
    param_grid = conf["decision-tree"]["param_grid"]

    # Use recall as the scoring metric
    scorer = make_scorer(recall_score, pos_label=0)
    
    grid_search = GridSearchCV(decision_tree, param_grid, cv=conf["training"]["cv"], scoring=scorer)

    grid_search.fit(X_train, y_train)

    return decision_tree, grid_search.best_params_
    
def DT_test(X_train, y_train, X_test, y_test, best_params):

    # Additional parameters (you may have other parameters like class_weight)
    additional_params = {
        'class_weight': conf["decision-tree"]["class_weights"],
        'random_state': 42
    }

    # Combine best_params and additional_params to create the final parameters
    final_params = {**best_params, **additional_params}

    # Create a decision tree classifier with the best parameters
    decision_tree = DecisionTreeClassifier(**final_params)
    
    # Train the model on the training data
    decision_tree.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = decision_tree.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    return decision_tree, accuracy, report, y_pred, y_test

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
    
    

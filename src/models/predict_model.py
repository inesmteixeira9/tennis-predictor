"""

python src/models/predict_model.py
"""
import click
import logging
import sys
sys.path.append('src/data/')
from make_dataset import load_data
from make_dataset import save_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml 
import pickle
from pathlib import Path

""" Make predictions
"""

print("Note: If the odds are equal for both players consider P1 to be the player with the highest ranking.")

odd1 = input("What's the 1st player odd?  ")
while not odd1.replace('.', '', 1).isdigit():
    print("Please enter a valid number with decimal points.")
    odd1 = input("What's the 1st player odd?  ")

odd2 = input("What's the 2nd player odd?  ")
while not odd2.replace('.', '', 1).isdigit():
    print("Please enter a valid number with decimal points.")
    odd2 = input("What's the 2nd player odd?  ")

rank1 = input("What's the 1st player rank?  ")
while not rank1.isdigit():
    print("Please enter a valid number.")
    rank1 = input("What's the 1st player rank?  ")

rank2 = input("What's the 2nd player rank?  ")
while not rank2.isdigit():
    print("Please enter a valid number.")
    rank2 = input("What's the 2nd player rank?  ")

if odd1 < odd2:
    p1_odd, p2_odd = odd1, odd2
    RankP1, RankP2 = rank1, rank2
elif odd1 == odd2:
    if rank1 < rank2:
        p1_odd, p2_odd = odd1, odd2
        RankP1, RankP2 = rank1, rank2
else:
    p1_odd, p2_odd = odd2, odd1
    RankP1, RankP2 = rank2, rank1

P1_wins = input("How many wins does P1 have over P2?  ")
while not P1_wins.isdigit():
    print("Please enter a valid number.")
    P1_wins = input("How many wins does P1 have over P2?  ")

P2_wins = input("How many wins does P2 have over P1?  ")
while not P2_wins.isdigit():
    print("Please enter a valid number.")
    P2_wins = input("How many wins does P2 have over P1?  ")
    
h2h = int(P1_wins) - int(P2_wins)

input = {
'OddP1': [p1_odd],
'RankP2': [int(RankP2)],
'H2H': [int(h2h)]
}

input = pd.DataFrame(input)

with open("C:/Users/inesm/projectos/tennis-predictor/conf/conf.yaml", "r") as config_file:
    conf = yaml.safe_load(config_file)

input_filepath = "C:/Users/inesm/projectos/tennis-predictor/data/processed/"

# Load data using the load_data function
dataset = load_data(input_filepath, file_name='features_classified.pkl')

# load auxiliar data to use in the custom profit function
aux_dataset = load_data(input_filepath, file_name='features.pkl')

# Define training datasets
y = dataset["winner_is_p1"]    
features_names = conf['classified_features']
X = dataset[features_names]

# classifier best parameters
best_params = conf['random-forest']['best-params']

# Additional parameters
additional_params = {
    'class_weight': conf['training']['class_weights'],
    'random_state': 42
}

# Combine best_params and additional_params to create the final parameters
final_params = {**best_params, **additional_params}

# Create a decision tree classifier with the best parameters
classifier = RandomForestClassifier(**final_params)

# Train the model on the training data
classifier.fit(X, y)

# Make predictions on the test data
y_pred = classifier.predict(input)

if y_pred == 1:
  print('Bet on P1')
else:
  print('Bet on P2')


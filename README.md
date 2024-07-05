# Tennis Predictor

## Project Overview

This project aims to provide betting recommendations for tennis matches based on historical match data. By analyzing various features such as player rankings, head-to-head records, surface type, and recent performance, the model predicts the outcome of tennis matches and suggests betting strategies to maximize profit.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

## Data Collection

The data for this project is sourced from the following locations:
- [ATP Match Data](https://data.world/tylerudite/atp-match-data)
- [WTA Match Data](https://data.world/tylerudite/wta-match-data)

## Project Structure

The project directory structure is organized as follows:


```python
tennis-betting-recommendation-system/
│
├── data/
│   ├── raw/
│   │   ├── atp/
│   │   └── wta/
│   ├── interim/
│   └── processed/
│
├── app/
│   ├── main.py
│   ├── build_features.py
│   ├── config.py
│   ├── libs/
│   │   ├── data_utils.py
│   │   ├── eda_tools.py
│   │   └── monitoring.py
│   ├── Makefile
│   └── requirements.txt
│
└── notebooks/
    ├── 1.0-data-exploration.ipynb
    └── 2.0-feature-engineering.ipynb

```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/inesmteixeira/tennis-predictor.git
    cd tennis-predictor
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the FastAPI server and use the endpoints, follow these steps:

1. Start the FastAPI server:
    ```sh
    uvicorn app.server:app --reload
    ```

2. Use the following endpoints to interact with the system:

## API Endpoints

### Upload CSV File

- **Endpoint**: `/extract/csvfiles`
- **Method**: POST
- **Description**: Upload a CSV file containing raw match data.
- **Parameters**: 
  - `csv_file`: The CSV file to upload.

### Extract Features

- **Endpoint**: `/get_features`
- **Method**: GET
- **Description**: Extract features from the raw data and prepare it for training.

### Train Model

- **Endpoint**: `/train`
- **Method**: GET
- **Description**: Train the model using the processed data.
- **Parameters**: 
  - `data`: JSON object containing the input data required for training.

### Predict Outcome

- **Endpoint**: `/predict`
- **Method**: GET
- **Description**: Predict the outcome of a tennis match.
- **Parameters**: 
  - `data`: JSON object containing the input data required for prediction.

## Features
### Data Transformation
The transform_data.py script includes the following functions to preprocess and transform the data:

- **transform_data**: Reads raw ATP and WTA data, cleans and formats it, and combines the datasets.
- **add_ranks**: Adds player rankings to the dataset.
- **add_rank_dif**: Adds the difference in rankings between players.
- **add_odd_dif**: Adds the difference in betting odds between players.
- **add_rank_ratio**: Adds the ratio of rankings between players.
- **add_odd_ratio**: Adds the ratio of betting odds between players.
- **OHE_surface**: One-hot encodes the surface type.
- **add_consecutive_wins_and_losses**: Calculates consecutive wins and losses for each player.

### Model Training and Prediction
- **main.py**: The entry point for the project. This script loads the data, transforms it using the functions in transform_data.py, and generates betting recommendations based on the model's predictions.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Submit a pull request with a description of your changes.

from pipeline import logging
from pipeline.clean_data import DataCleaner
from pipeline.features.build_features import FeaturesBuilder
from pipeline.models.logistic_regression import LogisticRegressionTrainer
import logging, argparse

STAGE_NAME_01 = 'Data Preparation'
STAGE_NAME_02 = 'Model Training'

def preparing_data():
    """
    Executes the data preparation stage, including data cleaning and feature building.
    """
    try:
        logging.info(f">>>>>> Stage: {STAGE_NAME_01} started <<<<<<")
        
        # Data cleaning
        cleaned_data = DataCleaner.main()
        
        # Feature building
        features = FeaturesBuilder.main(cleaned_data)
        
        logging.info(f">>>>>> Stage: {STAGE_NAME_01} completed <<<<<<\n\nx==========x")
        return features
    except Exception as e:
        logging.error(f"Error during {STAGE_NAME_01}: {e}", exc_info=True)
        raise e


def model_training():
    """
    Executes the model training stage, including initializing and training the model.

    Parameters:
        config (dict, optional): Configuration parameters for the Logistic Regression Trainer.
    """
    try:
        logging.info(f">>>>>> Stage: {STAGE_NAME_02} started <<<<<<")
        
        # Initialize the trainer
        logistic_regression = LogisticRegressionTrainer()
        
        # Create and train the model
        model = logistic_regression.create_model()
        trained_model = logistic_regression.train_model(model)
        logistic_regression.evaluate(trained_model)
        logistic_regression.save_model(trained_model)
        
        logging.info(f">>>>>> Stage: {STAGE_NAME_02} completed <<<<<<\n\nx==========x")
        return model
    except Exception as e:
        logging.error(f"Error during {STAGE_NAME_02}: {e}", exc_info=True)
        raise e

def main(stage_name=None):
    """
    Main function to execute the specified pipeline stage(s) using a steps approach.

    Parameters:
        stage_name (str, optional): Name of the stage to execute. If None, all stages are executed in order.
    """
    # Define pipeline steps in sequence
    steps = {
        STAGE_NAME_01: preparing_data,
        STAGE_NAME_02: model_training,
    }

    try:
        if stage_name:
            # Run only the specified stage
            if stage_name in steps:
                logging.info(f"Executing stage: {stage_name}")
                steps[stage_name]()
            else:
                logging.error(f"Invalid stage name: {stage_name}. Valid options: {list(steps.keys())}")
        else:
            # Run all stages sequentially
            logging.info("Executing all pipeline stages sequentially.")
            for step_name, step_function in steps.items():
                logging.info(f"Starting stage: {step_name}")
                step_function()

    except Exception as e:
        logging.error(f"Pipeline execution failed at stage '{stage_name}': {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Pipeline Stage Execution")
        parser.add_argument('--stage_name', required=False, help='Stage name to execute (e.g., "Data Preparation", "Model Training"). Leave empty to run all stages.')
        args = parser.parse_args()
        
        # Run the pipeline
        main(stage_name=args.stage_name)
    except SystemExit as e:
        logging.error(
            f"An error occurred while parsing arguments or executing the script: {e}. "
            f"Usage: python script.py --stage_name 'Data Preparation'",
            exc_info=True
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
import argparse
import joblib
import logging
from . import PARAMS

# Define the predict function
def predict(rank_p1, rank_p2, odd_p1, odd_p2, surface, h2h, consecutive_wins_p1, consecutive_wins_p2, 
            consecutive_losses_p1, consecutive_losses_p2, rank_evol_p1, rank_evol_p2):
    # Load the model
    model_path = PARAMS.logistic_regression.model_path
    model = joblib.load(model_path)

    # Get features
    features = [
        rank_p1,
        rank_p2,
        odd_p1,
        odd_p2,
        surface,
        h2h,
        consecutive_wins_p1,
        consecutive_wins_p2,
        consecutive_losses_p1,
        consecutive_losses_p2,
        rank_evol_p1,
        rank_evol_p2
    ]

    # Predict and return the result
    if model.predict([features]) == 1:
        return "Bet on player 1"
    else:
        return "Bet on player 2"

# Command-line argument parsing and script execution
if __name__ == "__main__":
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Tennis Match Prediction")
        parser.add_argument('--rank_p1', type=int, required=True, help='Player 1 rank')
        parser.add_argument('--rank_p2', type=int, required=True, help='Player 2 rank')
        parser.add_argument('--odd_p1', type=float, required=True, help='Player 1 odds')
        parser.add_argument('--odd_p2', type=float, required=True, help='Player 2 odds')
        parser.add_argument('--surface', type=int, required=True, choices=[0, 1, 2], 
                            help='Surface type: 0 for Hard, 1 for Clay, 2 for Grass')
        parser.add_argument('--h2h', type=int, required=True, help='Head-to-Head wins for player 1')
        parser.add_argument('--consecutive_wins_p1', type=int, required=True, help='Consecutive wins for player 1')
        parser.add_argument('--consecutive_wins_p2', type=int, required=True, help='Consecutive wins for player 2')
        parser.add_argument('--consecutive_losses_p1', type=int, required=True, help='Consecutive losses for player 1')
        parser.add_argument('--consecutive_losses_p2', type=int, required=True, help='Consecutive losses for player 2')
        parser.add_argument('--rank_evol_p1', type=int, required=True, help='Rank evolution for player 1')
        parser.add_argument('--rank_evol_p2', type=int, required=True, help='Rank evolution for player 2')
        
        args = parser.parse_args()

        # Call the predict function with the parsed arguments
        result = predict(
            args.rank_p1, args.rank_p2, args.odd_p1, args.odd_p2,
            args.surface, args.h2h, args.consecutive_wins_p1, args.consecutive_wins_p2,
            args.consecutive_losses_p1, args.consecutive_losses_p2,
            args.rank_evol_p1, args.rank_evol_p2
        )

        # Print the result of the prediction
        print(result)

    except SystemExit as e:
        logging.error(
            f"An error occurred while parsing arguments or executing the script: {e}. "
            f"Usage: python script.py --rank_p1 <value> --rank_p2 <value> ...",
            exc_info=True
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)

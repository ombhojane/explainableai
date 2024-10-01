import colorama
from custom_logger import setup_logger

# Initialize colorama
colorama.init()

# Initialize logger
logger = setup_logger('ModelAnalysis')

def analyze_model(accuracy):
    logger.info("Analyzing model...")
    logger.info(f"Accuracy: {accuracy:.3f}")

def display_error(message):
    logger.error(f"Error: {message}")

def display_warning(message):
    logger.warning(f"Warning: {message}")

def main():
    accuracy = 0.91234  # Example accuracy value
    analyze_model(accuracy)
    
    # Example usage of other log functions
    display_error("An error occurred while loading the model.")
    display_warning("This model may take a long time to train.")

if __name__ == "__main__":
    main()

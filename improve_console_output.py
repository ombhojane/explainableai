import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)  # Ensures colors are reset automatically after each print

def analyze_model(accuracy):
    # Enhanced print formatting for analysis messages
    print(f"{Fore.BLUE}Analyzing model...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Accuracy: {accuracy:.3f}")

def display_error(message):
    # Use red for error messages
    print(f"{Fore.RED}Error: {message}")

def display_warning(message):
    # Use yellow for warning messages
    print(f"{Fore.YELLOW}Warning: {message}")

def main():
    accuracy = 0.91234  # Example accuracy value
    analyze_model(accuracy)
    
    # Example usage of enhanced error and warning messages
    display_error("An error occurred while loading the model.")
    display_warning("This model may take a long time to train.")

if __name__ == "__main__":
    main()

from colorama import Fore, Style

def analyze_model(accuracy):
    print(f"{Fore.BLUE}Analyzing model...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Accuracy: {accuracy:.3f}{Style.RESET_ALL}")

def display_error(message):
    print(f"{Fore.RED}Error: {message}{Style.RESET_ALL}")

def display_warning(message):
    print(f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}")

def main():
    accuracy = 0.91234  # Example accuracy value
    analyze_model(accuracy)
    
    # Example usage of other print functions
    display_error("An error occurred while loading the model.")
    display_warning("This model may take a long time to train.")

if __name__ == "__main__":
    main()

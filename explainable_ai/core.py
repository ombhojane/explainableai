import time
from .utils import explain_model

def xai_wrapper(train_function):
    def wrapper(*args, **kwargs):
        print("Starting XAI wrapper...")
        
        start_time = time.time()
        model = train_function(*args, **kwargs)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Model training completed in {training_time:.2f} seconds.")
        
        explanation = explain_model(model)
        
        return model, explanation
    
    return wrapper
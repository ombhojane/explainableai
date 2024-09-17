# explainable_ai/llm_explanations.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def initialize_gemini():
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    return genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config, safety_settings=safety_settings)

def get_llm_explanation(model, results):
    prompt = f"""
    As an AI expert, please provide a clear and concise explanation of the following machine learning model results for a non-technical audience:

    Model Performance:
    {results['model_performance']}

    Top 5 Important Features:
    {dict(list(results['feature_importance'].items())[:5])}

    Cross-validation Score:
    Mean: {results['cv_scores'][0]:.4f}
    Standard Deviation: {results['cv_scores'][1]:.4f}

    Please explain:
    1. What these results mean in simple terms.
    2. The model's overall performance and reliability.
    3. Which features are most important and why they might matter.
    4. Suggestions for potential next steps or areas of improvement.

    Format your response as follows:
    
    Summary:
    [Provide a brief 2-3 sentence summary of the overall results]

    Model Performance:
    [Explain the model's performance metrics]

    Important Features:
    [Discuss the top 5 important features and their potential significance]

    Next Steps:
    [Suggest 2-3 potential next steps or areas for improvement]

    Keep the explanation under 500 words and avoid technical jargon.
    """

    response = model.generate_content(prompt)
    return response.text


def get_prediction_explanation(model, input_data, prediction, probabilities, feature_importance):
    top_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    
    prompt = f"""
    As an AI expert, please provide a clear and concise explanation of the following prediction for a non-technical audience:

    Input Data:
    {input_data}

    Prediction: {prediction}
    Probabilities: {probabilities}

    Top 5 Important Features:
    {top_features}

    Please explain:
    1. What the prediction means in simple terms.
    2. How confident the model is in its prediction.
    3. Which input features likely contributed most to this prediction and why.
    4. Any potential limitations or considerations for this prediction.
    5. Suggestions for what the user might do with this information.

    Format your response as follows:
    
    Prediction Summary:
    [Provide a brief 2-3 sentence summary of the prediction and its confidence]

    Key Factors:
    [Discuss the top 3-5 input features that likely influenced this prediction]

    Considerations:
    [Mention any limitations or important considerations for interpreting this prediction]

    Next Steps:
    [Suggest 2-3 potential actions or decisions the user might make based on this prediction]

    Keep the explanation under 300 words and avoid technical jargon.
    """

    response = model.generate_content(prompt)
    return response.text
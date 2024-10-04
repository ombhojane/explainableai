import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducibility
np.random.seed(42)

# Generate 100 rows of data
data_size = 100

# Generate features
logger.info("Generating random data for features...")
age = np.random.randint(18, 90, data_size)
tumor_size = np.random.uniform(1, 10, data_size)
gene_mutation = np.random.choice([0, 1], data_size)
smoking_history = np.random.randint(0, 40, data_size)

# Vectorized computation of cancer risk
cancer = np.where((age > 50) & (tumor_size > 5) & (gene_mutation == 1) & (smoking_history > 20), 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Tumor_Size': tumor_size,
    'Gene_Mutation': gene_mutation,
    'Smoking_History': smoking_history,
    'Cancer': cancer
})

# Save the DataFrame to a CSV file
output_file = 'cancer.csv'
logger.info(f"Saving the CSV file '{output_file}'...")
df.to_csv(output_file, index=False)
logger.info(f"CSV file '{output_file}' has been created.")
import numpy as np
import pandas as pd
import logging


logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set seed for reproducibility
logger.debug("Setting the speed...")
np.random.seed(42)

# Generate 100 rows of data
logger.debug("Setting up the data...")
data_size = 100

# Generate 4 features with random values
# Feature 1: Age (18 to 90)
logger.debug("Generating the features...")
age = np.random.randint(18, 90, data_size)

# Feature 2: Tumor size (1 to 10 cm)
tumor_size = np.random.uniform(1, 10, data_size)

# Feature 3: Gene mutation presence (0 or 1)
gene_mutation = np.random.choice([0, 1], data_size)

# Feature 4: Smoking history (years, 0 to 40)
smoking_history = np.random.randint(0, 40, data_size)

# Generate target column 'Cancer' based on some logic
# For simplicity, assume higher risk if age > 50, tumor_size > 5, gene_mutation = 1, and smoking_history > 20
logger.debug("Generating the column...")
cancer_risk = (age > 50) & (tumor_size > 5) & (gene_mutation == 1) & (smoking_history > 20)
cancer = np.where(cancer_risk, 1, 0)

# Create a DataFrame
logger.debug("Creating the DataFrame...")
df = pd.DataFrame({

    'Age': age,
    'Tumor_Size': tumor_size,
    'Gene_Mutation': gene_mutation,
    'Smoking_History': smoking_history,
    'Cancer': cancer
})

# Save the DataFrame to a CSV file
logger.info("Saving the csv file...")
df.to_csv('cancer.csv', index=False)

logger.info("CSV file 'cancer.csv' has been created.")

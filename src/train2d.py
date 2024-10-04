import os  # Add this import
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import deepchem as dc # type: ignore
from deepchem.models.wandblogger import WandbLogger  # type: ignore
import wandb # type: ignore
import numpy as np # type: ignore

# Initialize WandbLoggers
logger = WandbLogger(project="PAGTN-Molecule")

# Load the dataset
print("Loading dataset...")
train_file = "data/raw/data-GCxGC_train.csv"
val_file = "data/raw/data-GCxGC_val.csv"
prefix = train_file.split("/")[-1].split(".")[0].split("_")[0]

# Check if pre-saved data exists
featurized_train_file = f"data/featurized/{prefix}/train_features.npy"
featurized_val_file = f"data/featurized/{prefix}/val_features.npy"
train_labels_file = f"data/featurized/{prefix}/train_labels.npy"
val_labels_file = f"data/featurized/{prefix}/val_labels.npy"

# Featurizer
print("Initializing Featurizer...")
featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)

try:
    # Try loading previously saved data
    print("Loading saved featurized data and labels...")
    train_X = np.load(featurized_train_file)
    val_X = np.load(featurized_val_file)
    train_labels = np.load(train_labels_file)  # Shape: (num_samples, 2)
    val_labels = np.load(val_labels_file)      # Shape: (num_samples, 2)
except FileNotFoundError:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Extract SMILES strings and labels
    print("Extracting SMILES strings and labels...")
    train_smiles = train_df['smiles'].tolist()
    train_labels = train_df[['ri', 'ri_2']].values  # Extract both 'ri' and 'ri_2'
    val_smiles = val_df['smiles'].tolist()
    val_labels = val_df[['ri', 'ri_2']].values      # Extract both 'ri' and 'ri_2'
        
    # Featurize the data
    print("Featurizing training SMILES strings...")
    train_X = featurizer.featurize(train_smiles)
    print("Featurizing validation SMILES strings...")
    val_X = featurizer.featurize(val_smiles)

    # Ensure the target directory exists
    featurized_dir = os.path.dirname(featurized_train_file)
    os.makedirs(featurized_dir, exist_ok=True)
    print(f"Ensured the directory '{featurized_dir}' exists.")

    # Save the featurized data and labels
    print("Saving featurized data and labels...")
    np.save(featurized_train_file, train_X)
    np.save(featurized_val_file, val_X)
    np.save(train_labels_file, train_labels)  # Shape: (num_samples, 2)
    np.save(val_labels_file, val_labels)      # Shape: (num_samples, 2)


# Create datasets
print("Creating datasets...")
train_dataset = dc.data.NumpyDataset(X=train_X, y=np.array(train_labels))
val_dataset = dc.data.NumpyDataset(X=val_X, y=np.array(val_labels))

# Define the metric for validation
print("Defining validation metric...")
metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")

# Create the validation callback
print("Creating validation callback...")
val_callback = dc.models.ValidationCallback(val_dataset, 500, [metric])

# Initialize the PAGTN model with two tasks
print("Initializing PAGTN model...")
model = dc.models.PagtnModel(
    n_tasks=2,  # Number of tasks (for both 'ri' and 'ri_2')
    mode='regression',
    dropout=0.2,  # Dropout rate
    learning_rate=5e-4,  # Learning rate
    batch_size=50,  # Batch size
    num_layers=8,  # Number of layers
    num_heads=1,  # Number of heads
    hidden_features=280,  # Hidden features
    output_node_features=280,  # Output node features
    wandb_logger=logger
)

# Train the model
print("Training the model...")
model.fit(train_dataset, nb_epoch=300, callbacks=[val_callback])

# Evaluate the model
print("Evaluating the model...")
train_score = model.evaluate(train_dataset, metrics=[metric])
val_score = model.evaluate(val_dataset, metrics=[metric])

print(f'Train MAE: {train_score["mean_absolute_error"]}')
print(f'Validation MAE: {val_score["mean_absolute_error"]}')

# Log scores to WandB
print("Logging scores to WandB...")
columns = ["train", "val"]
metrics_table = wandb.Table(columns=columns)
metrics_table.add_data(train_score, val_score)
logger.wandb_run.log({"Scores": metrics_table})

# Finish logging
logger.finish()

print("Program completed successfully!")

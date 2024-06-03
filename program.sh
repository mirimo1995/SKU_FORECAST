python -m venv .venv
source .venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the preprocessing step
python scripts/preprocess.py --data_path data.csv --output_path . --lead_time 30

# Run the training step
python scripts/train.py --train_data_path train_data.csv --test_data_path test_data.csv --output_model_path output.model --metrics_output_path metrics.json

deactivate
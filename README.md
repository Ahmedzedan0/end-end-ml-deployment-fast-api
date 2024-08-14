# End-to-End Machine Learning Deployment with FastAPI

This repository contains a full end-to-end machine learning deployment setup using FastAPI. The project demonstrates how to train a machine learning model, deploy it as an API using FastAPI, and manage the data and model versioning with DVC (Data Version Control).

## Project Structure

```
end-end-ml-deployment-fast-api/
├── 📁.dvc/                        # DVC configuration and cache files
├── 📁.git/                        # Git configuration and history
├── 📁data/                        # Dataset files and DVC tracking
│   ├── census.csv                 # Main dataset file
│   └── census.csv.dvc             # DVC file for dataset versioning
├── 📁model/                       # Trained model and encoders
│   ├── encoder.joblib             # Label encoder
│   ├── lb.joblib                  # Label binarizer
│   └── random_forest_model.joblib # Trained Random Forest model
├── 📁screenshots/                 # Screenshots for documentation
│   ├── fast_api_doc.png           # FastAPI documentation screenshot
│   ├── get.png                    # Example of GET request screenshot
│   ├── post.png                   # Example of POST request screenshot
│   ├── render.png                 # Example of rendered page screenshot
│   └── responses.png              # Example of API responses screenshot
├── 📁starter/                     # Source code for data processing and model training
│   ├── 📁ml/                      # Machine learning scripts
│   │   ├── __init__.py            # Init file for ML module
│   │   ├── data.py                # Data processing functions
│   │   ├── model.py               # Model training and inference functions
│   │   └── test_model.py          # Unit tests for the model
│   ├── __init__.py                # Init file for starter module
│   └── train_model.py             # Script to train the model
├── .dvcignore                     # Files and directories ignored by DVC
├── environment.yml                # Conda environment configuration
├── folder_structure               # Visual representation of the project structure
├── main.py                        # Main FastAPI application script
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── sanitycheck.py                 # Script for checking environment setup
├── setup.py                       # Setup script for packaging
└── test_main.py                   # Unit tests for the FastAPI application
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/end-end-ml-deployment-fast-api.git
   cd end-end-ml-deployment-fast-api
   ```

2. **Set Up the Environment:**

   Create a conda environment using the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```

   Alternatively, install dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**

   Train the machine learning model by running the `train_model.py` script:

   ```bash
   python starter/train_model.py
   ```

4. **Run the FastAPI Application:**

   Start the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

5. **API Documentation:**

   Once the server is running, visit `http://127.0.0.1:8000/docs` to see the automatically generated API documentation.

## Data Version Control

This project uses DVC for dataset and model versioning. To track changes and manage your data:

- **Add Data to DVC:**

  ```bash
  dvc add data/census.csv
  ```

- **Commit and Push Changes:**

  ```bash
  git add .
  git commit -m "Add dataset"
  dvc push
  ```

## Testing

Run unit tests for the application:

```bash
pytest
```

## Screenshots

Screenshots of the API responses and documentation are available in the `screenshots/` directory.

---
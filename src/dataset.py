from pathlib import Path

import pandas as pd
import ssl

from src.config import EXTERNAL_DATA_DIR
from ucimlrepo import fetch_ucirepo

def download():
    
    # ! Not safe in production
    ssl._create_default_https_context = ssl._create_unverified_context
        
    heart_failure_clinical_records = fetch_ucirepo(id=519)

    features_df = heart_failure_clinical_records.data.features
    features_df.to_csv( EXTERNAL_DATA_DIR / "features.csv", index=False)

    targets_df = heart_failure_clinical_records.data.targets
    targets_df.to_csv(EXTERNAL_DATA_DIR / "targets.csv", index=False)
    
    heart_failure_df = pd.concat([targets_df, features_df], axis=1)
    heart_failure_df.to_csv(EXTERNAL_DATA_DIR / "heart_failure.csv", index=False)
            
if __name__ == "__main__":
    download()

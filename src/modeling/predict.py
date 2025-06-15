from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import torch
import gpytorch

from src.modeling.model import ExactGPModel

app = typer.Typer()

@app.command("predict")
def predict():
    train_x = torch.load('train_x.pth')
    train_y = torch.load('train_y.pth')

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.load_state_dict(torch.load('model.pth'))
    likelihood.load_state_dict(torch.load('likelihood.pth'))
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.randint(20, 101, (51, 1), dtype=torch.float32)
        observed_pred = likelihood(model(test_x))
        
    torch.save({
    "mean": observed_pred.mean,
    "lower": observed_pred.confidence_region()[0],
    "upper": observed_pred.confidence_region()[1],
    "test_x": test_x
    }, "observed_pred.pth")

if __name__ == "__main__":
    app()

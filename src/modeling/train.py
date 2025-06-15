import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt

from src.config import EXTERNAL_DATA_DIR
from src.modeling.model import ExactGPModel

def get_data():
    data = pd.read_csv(EXTERNAL_DATA_DIR / "heart_failure.csv")
    training_data = data[['ejection_fraction', 'age']]
    return training_data
  
def train():
    training_data = get_data()
    
    train_x = torch.tensor(training_data['age'].values, dtype=torch.float32).unsqueeze(1)
    train_y = torch.tensor(training_data['ejection_fraction'].values)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    
    training_iter = 50

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        
        optimizer.step()
        
    torch.save(model.state_dict(), 'model.pth')
    torch.save(likelihood.state_dict(), 'likelihood.pth')
    torch.save(train_x, 'train_x.pth')
    torch.save(train_y, 'train_y.pth')
        
        
if __name__ == "__main__":
    train()


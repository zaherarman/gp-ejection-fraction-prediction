import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt

from src.config import EXTERNAL_DATA_DIR
from src.modeling.model import ExactGPModel
from sklearn.model_selection import train_test_split

def get_data():
    data = pd.read_csv(EXTERNAL_DATA_DIR / "heart_failure.csv")
    return data[['ejection_fraction', 'age']]
     
def train():
    training_data = get_data()
    
     # Split real data into train/test (80/20)
    train_x, test_x, train_y, test_y = train_test_split(
        training_data[['age']].values, 
        training_data['ejection_fraction'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)


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
    torch.save(test_x, 'test_x.pth')
    torch.save(test_y, 'test_y.pth')
        
if __name__ == "__main__":
    train()


import torch
import gpytorch

from src.modeling.model import ExactGPModel

def predict():
    train_x = torch.load('train_x.pth')
    train_y = torch.load('train_y.pth') 
    test_x = torch.load("test_x.pth")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.load_state_dict(torch.load('model.pth'))
    likelihood.load_state_dict(torch.load('likelihood.pth'))
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        
    torch.save({
        "mean": observed_pred.mean,
        "lower": observed_pred.confidence_region()[0],
        "upper": observed_pred.confidence_region()[1],
        "test_x": test_x
    }, "observed_pred.pth")

if __name__ == "__main__":
    predict()

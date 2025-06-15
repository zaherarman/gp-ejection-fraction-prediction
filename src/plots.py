import torch
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

def plot_gp_predictions():
    # Load training data
    train_x = torch.load("train_x.pth")
    train_y = torch.load("train_y.pth")

    # Load predictions
    pred_data = torch.load("observed_pred.pth")
    mean = pred_data["mean"]
    lower = pred_data["lower"]
    upper = pred_data["upper"]
    test_x = pred_data["test_x"]

    # Flatten and convert to numpy
    x = test_x.squeeze().numpy()
    mean_np = mean.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()

    # Sort by x for proper plotting
    sorted_indices = x.argsort()
    x_sorted = x[sorted_indices]
    mean_sorted = mean_np[sorted_indices]
    lower_sorted = lower_np[sorted_indices]
    upper_sorted = upper_np[sorted_indices]

    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(x_sorted, mean_sorted, 'b')
        ax.fill_between(x_sorted, lower_sorted, upper_sorted, alpha=0.5)

        # Dynamically set y-limits based on data
        y_min = min(train_y.min().item(), lower.min().item())
        y_max = max(train_y.max().item(), upper.max().item())
        padding = 0.1 * (y_max - y_min)
        ax.set_ylim([y_min - padding, y_max + padding])

        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.savefig(FIGURES_DIR / "gp_prediction_plot.png")
        plt.close(f)

if __name__ == "__main__":
    plot_gp_predictions()

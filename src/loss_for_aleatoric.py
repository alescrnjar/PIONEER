
import torch
import torch.nn as nn

def gaussian_nll_loss(y_true, y_pred):
    mean = y_pred[:, 0].unsqueeze(1)
    log_variance = y_pred[:, 1].unsqueeze(1)

    # Calculate the negative log-likelihood
    mse = nn.functional.mse_loss(y_true, mean, reduction='none')
    variance = torch.exp(log_variance)
    pi = torch.tensor(3.141592653589)
    nll = 0.5 * (torch.log(2 * pi * variance) + mse / variance)

    # Return the average NLL across the batch
    return nll.mean()
"""
def gaussian_nll_loss(y_true, y_pred):
    mean = tf.expand_dims(y_pred[:,0], axis=1)
    log_variance = tf.expand_dims(y_pred[:,1], axis=1)

    # Calculate the negative log-likelihood
    mse = keras.losses.mean_squared_error(y_true, mean)
    variance = tf.exp(log_variance)
    pi = tf.constant(3.141592653589)
    nll = 0.5 * (tf.math.log(2 * pi * variance) + mse / variance)

    # Return the average NLL across the batch
    return tf.reduce_mean(nll)
"""


if __name__=='__main__':
    # Example usage
    y_true = torch.tensor([[1.0], [2.0]])
    y_pred = torch.tensor([[1.5, -0.5], [2.5, -0.5]])

    loss = gaussian_nll_loss(y_true, y_pred)
    print(loss)
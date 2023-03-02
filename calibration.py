import torch
from torch.autograd import Variable

def calibrate(x, y, alpha_ref, beta_ref, learning_rate=1e-4, epochs=4000, r=1):

    dtype = torch.FloatTensor

    x = torch.from_numpy(x).type(dtype)
    y = torch.from_numpy(y).type(dtype)

    # clip x, y to shortest one
    x = x[:,:min(x.shape[1], y.shape[1])]
    y = y[:,:min(x.shape[1], y.shape[1])]

    alpha_ref = torch.from_numpy(alpha_ref)
    beta_ref = torch.from_numpy(beta_ref)

    alpha = Variable(alpha_ref.type(dtype), requires_grad=True)
    beta = Variable(beta_ref.type(dtype), requires_grad=True)

    # calibrate alpha, beta using backprop
    
    for t in range(epochs):
        
        # forward pass
        y_pred = (x - beta) * (3300.0) / (1023.0 * alpha)
        
        # compute loss
        loss = (y_pred - y).pow(2).sum() + r * ((alpha - alpha_ref).pow(2).sum() + (beta - beta_ref).pow(2).sum())
        
        # backprop
        loss.backward()
        
        # update parameters
        alpha.data -= learning_rate * alpha.grad.data
        beta.data -= learning_rate * beta.grad.data
        
        # zero gradients
        alpha.grad.data.zero_()
        beta.grad.data.zero_()
            
        if t % 500 == 0:
            print("t: ", t, " loss: ", loss.data)
        
    return alpha, beta

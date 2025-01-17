import matplotlib.pyplot as plt
import matplotlib
import torch

def lossf(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


def predict(net, x, y, title):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o',
             c='r', label='Prediction')
    plt.title(title)
    plt.show(block=True)

x_train = torch.rand(100)
x_train = x_train*20 - 10.0
y_train = torch.sin(x_train)
plt.plot(x_train.numpy(),y_train.numpy(),'o')
plt.show(block=True)
noise = torch.randn(y_train.shape)/5.0
plt.plot(x_train.numpy(),noise.numpy(),'o')
plt.title('Gaussian noise')
plt.show(block=True)
y_train = y_train+noise
plt.plot(x_train.numpy(),y_train.numpy(),'o')
plt.title('noisy sine')
plt.show(block=True)
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)
x_validation = torch.linspace(-10,10,100)
y_validation = torch.sin(x_validation)
plt.figure(4)
plt.plot(x_validation.numpy(),y_validation.numpy(),'o')
plt.title('sin(x)')
plt.xlabel('x_validation')
plt.ylabel('y_validation')
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
plt.title('validation set')
plt.show(block=True)
class SineNet(torch.nn.Module):
     def __init__(self,n_hidden_neurons):
        super(SineNet,self).__init__()
        self.fc1 = torch.nn.Linear(1,n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons,1)
     def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

sn = SineNet(50)
print('sn created')

plt.figure(5)
predict(sn,x_validation,y_validation,'1st prediction attempt ')
optimizer = torch.optim.Adam(sn.parameters(),lr=0.01)

##### Training procedure
loss_val = torch.ones(1)*10
for epoch_index in range(500):
     print('Epoch===================== ',epoch_index,loss_val.item())
     optimizer.zero_grad()
     y_pred = sn.forward(x_train)
     loss_val = lossf(y_pred,y_train)
     loss_val.backward()
     optimizer.step()
 #print(epoch_index)
print('FINAL LOSS VALUE ',loss_val)
plt.figure(6)
predict(sn,x_validation,y_validation,'prediction after training')
plt.show(block=True)


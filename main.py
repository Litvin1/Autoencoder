# Vadim Litvinov, Yaron Geffen
import torch
from torch import nn, optim
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
from torchgen.context import F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import pandas as pd

NUM_IMAGES = 124
EPOCHS = 10
BOTTLENECK_LAYER_DIM = 2
BATCH_SIZE = 32
TSNE_DIMS = 2
pos_labels = torch.ones(NUM_IMAGES)
neg_labels = torch.zeros(NUM_IMAGES)
labels = torch.cat([pos_labels, neg_labels])
# print(labels)
dataset = TensorDataset(tensor_images, labels)
data = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, bottleneck_layer_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2500, 1250)
        self.linear2 = nn.Linear(1250, 625)
        self.linear3 = nn.Linear(625, 312)
        self.linear4 = nn.Linear(312, 156)
        self.linear5 = nn.Linear(156, bottleneck_layer_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        # print(self.linear5(x))
        return self.linear5(x)


class Decoder(nn.Module):
    def __init__(self, bottleneck_layer_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(bottleneck_layer_dim, 156)
        self.linear2 = nn.Linear(156, 312)
        self.linear3 = nn.Linear(312, 625)
        self.linear4 = nn.Linear(625, 1250)
        self.linear5 = nn.Linear(1250, 2500)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.leaky_relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = self.linear5(z)
        return z.reshape((-1, 1, 50, 50))


class Autoencoder(nn.Module):
    def __init__(self, bottleneck_layer_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(bottleneck_layer_dim)
        self.decoder = Decoder(bottleneck_layer_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=EPOCHS):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs):
        gen_loss = 0
        for x, y in data:
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            gen_loss += loss
            loss.backward()
            opt.step()
        print(epoch, gen_loss)
    return autoencoder


autoencoder = Autoencoder(BOTTLENECK_LAYER_DIM).to(device)  # GPU/CPU
print(autoencoder)
autoencoder = train(autoencoder, data)


def get_bottleneck(autoencoder, bottleneck_layer_dim, tsne_dims, data, num_batches=100 / BATCH_SIZE):
    Z = []
    Y = []
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        for i, item in enumerate(z):
            Z.append(z[i].squeeze())
            Y.append(y[i].item())
    return Z, Y


Z, Y = get_bottleneck(autoencoder, BOTTLENECK_LAYER_DIM, TSNE_DIMS, data)
print(np.array(Z).shape)


def plot_latent(Z, Y, tsne_dims):
    tsne = TSNE(n_components=tsne_dims)
    # Z = np.append(Z, [Z0, 0], axis = 0)
    # plt.colorbar()
    # tsne_results = tsne.fit_transform(Z)
    df = pd.DataFrame(np.array(Z))
    df['y'] = Y
    print(np.any(np.isnan(Z)))
    print(np.all(np.isfinite(Z)))
    Z = np.nan_to_num(Z, copy=True)
    print(np.any(np.isnan(Z)))
    print(np.all(np.isfinite(Z)))
    tsne_results = tsne.fit_transform(Z)
    print(tsne_results.shape)
    plt.figure(figsize=(16, 10))
    plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='tab10')
    plt.colorbar()


plot_latent(Z, Y, TSNE_DIMS)


def plot_roc(tpr, fpr):
    plt.plot(fpr, tpr, label='roc', lw=1.5)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',
             label='Chance', alpha=.9)
    # mean can't be calculated because of different shape for every fold
    # fpr_mean = np.mean(np.array(fpr), axis=0)
    # tpr_mean = np.mean(tpr, axis=0)
    # plt.plot(fpr_mean, tpr_mean, label='mean', lw=2, color='red')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title('ROC curve')
    # plt.savefig('/home/ls/yarong/roc.jpeg')
    plt.show()
    plt.close()


# classify data
LEARNING_RATE = 0.00001


class nn_Network(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(nn_Network, self).__init__()
        self.encoder = encoder
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.encoder(x))
        x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.act(self.linear4(x))
        return x


network = nn_Network(autoencoder.encoder, BOTTLENECK_LAYER_DIM, 50, 25, 13, 1)
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=0.0001)
network.train()
criterion = nn.BCELoss()
for epoch in range(EPOCHS):
    outputs = []
    labels = []
    running_loss = 0.0
    correct = 0
    for batch_num, (inputs, label) in enumerate(data, start=1):
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()
        output = network(inputs)
        for elem in label:
            labels.append(elem.item())
        loss = criterion(output, label.view(BATCH_SIZE, -1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = (output > 0.5).float()
        for elem in output:
            outputs.append(elem.item())
        correct += (output[0] == label[0]).float().sum()
    tn, fp, fn, tp = confusion_matrix(labels, outputs).ravel()
    acc = 100 * correct / (len(data.dataset))
    loss = running_loss
    print(f"epoch num: {epoch} - Train loss: {loss:.3f}, Train accuracy: {acc:.3f}%")
    fpr, tpr, thresholds = metrics.roc_curve(labels, outputs)
    auc = metrics.auc(fpr, tpr)
    print(auc)
    plot_roc(tpr, fpr)


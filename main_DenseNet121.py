# Import the relevant packages
import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from MakeDataset import makedataset
from torchvision.models import densenet121
from utils import Flatten

# Set hyper-parameters
batchsz = 4
lr = 1e-3
epochs = 10

# Specify CUDA acceleration and set random seeds
device = torch.device('cuda')
torch.manual_seed(1234)

# load train/val/test datasets
train_db = makedataset('mydata', 224, mode='train')
val_db = makedataset('mydata', 224, mode='val')
test_db = makedataset('mydata', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)

# view numbers of datasets
print('num_train:', len(train_loader.dataset))
print('num_val:', len(val_loader.dataset))
print('num_test:', len(test_loader.dataset))

# Define validation functions
def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        # Calculate the correct number of predictions in the validation set
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total # Return accuracy

# Define main function training
def main():
    # Load the pre-training model
    trained_model = resnet18(pretrained=True)
    # Use the transfer learning replacement structure
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512, 6)
                          ).to(device)
    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    # Start training
    best_acc, best_epoch = 0, 0
    # The number of iterations
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Visual loss value
        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            print('Epoch:', epoch, '/', epochs - 1, 'acc_val:', val_acc, 'loss:', loss)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                print(best_acc)
                # Save the model weight parameter values
                torch.save(model.state_dict(), 'mydata/weights_resnet18.mdl')
    # Print the best training result
    print('best acc:', best_acc, 'best epoch:', best_epoch)


if __name__ == '__main__':
    main()
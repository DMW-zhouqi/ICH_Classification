# Import the relevant packages
import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from MakeDataset import makedataset
from torchvision.models import resnet18
from utils import Flatten

# Set the batch of the test set
batchsz = 4

# Specify CUDA acceleration and set random seeds
device = torch.device('cuda')
torch.manual_seed(1234)

# load test datasets
test_db = makedataset('mydata', 224, mode='test')
test_loader = DataLoader(test_db, batch_size=batchsz)
print('num_test:', len(test_loader.dataset)) # # view numbers of datasets

# Define test functions
def test(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    # Set the initial values for each subtype
    num_EDH, num_IVH, num_normal, num_CPH, num_SAH, num_SDH = 0, 0, 0, 0, 0, 0
    # Set the model to predict the initial values for each subtype
    num_EDH_T, num_IVH_T, num_normal_T, num_CPH_T, num_SAH_T, num_SDH_T = 0, 0, 0, 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for label in y:
            if label == 0:
                num_EDH += 1
            if label == 1:
                num_IVH += 1
            if label == 2:
                num_normal += 1
            if label == 3:
                num_CPH += 1
            if label == 4:
                num_SAH += 1
            if label == 5:
                num_SDH += 1
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            # The update model predicts the number of positives
            for i in range(len(pred)):
                for j in range(i, len(y)):
                    if pred[i] == y[j] and y[j] == 0:
                        num_EDH_T += 1
                    if pred[i] == y[j] and y[j] == 1:
                        num_IVH_T += 1
                    if pred[i] == y[j] and y[j] == 2:
                        num_normal_T += 1
                    if pred[i] == y[j] and y[j] == 3:
                        num_CPH_T += 1
                    if pred[i] == y[j] and y[j] == 4:
                        num_SAH_T += 1
                    if pred[i] == y[j] and y[j] == 5:
                        num_SDH_T += 1
                    break
        # Calculate the total number of correct Numbers
        correct += torch.eq(pred, y).sum().float().item()

    # visualization
    print('num_EDH:', num_EDH, 'num_EDH_T:', num_EDH_T)
    print('num_IVH:', num_IVH, 'num_IVH_T:', num_IVH_T)
    print('num_normal:', num_normal, 'num_normal_T:', num_normal_T)
    print('num_CPH:', num_CPH, 'num_CPH_T:', num_CPH_T)
    print('num_SAH:', num_SAH, 'num_SAH_T:', num_SAH_T)
    print('num_SDH:', num_SDH, 'num_SDH_T:', num_SDH_T)

    return correct / total # Return accuracy

# Define main function test
def main():
    # Load the pre-training model
    trained_model = resnet18(pretrained=True)
    # Use the transfer learning replacement structure
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          nn.Linear(512, 6)
                          ).to(device)
    # Start test
    test_acc = test(model, test_loader)
    # print final accuracy
    print('test acc:', test_acc)

if __name__ == '__main__':
    main()

import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from MakeDataset import makedataset
from torchvision.models import resnet18
from utils import Flatten
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

batchsz = 280 # Remove all test data at once
device = torch.device('cuda')
torch.manual_seed(1234)

test_db = makedataset('mydata', 224, mode='test')
test_loader = DataLoader(test_db, batch_size=batchsz)

# Do one-hot coding of labels
depth = 6
def one_hot(label, depth=6):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

# load model
trained_model = resnet18(pretrained=True)
model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                        Flatten(),  # [b, 512, 1, 1] => [b, 512]
                        nn.Linear(512, 6)
                        )
# load weights
model.load_state_dict(torch.load('mydata/weights_resnet18.mdl'))
print('loaded from ckpt!')

# plot ROC curve
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    y_onehot = one_hot(y, depth) # y.shape: [280, 6]
    with torch.no_grad():
        logits = model(x)

        # The horizontal and vertical coordinate values and AUC values of each category are calculated
        fpr_EDH, tpr_EDH, thresholds_EDH = roc_curve(y_onehot[:, 0], logits[:, 0])
        auc_EDH = auc(fpr_EDH, tpr_EDH)
        fpr_IVH, tpr_IVH, thresholds_IVH = roc_curve(y_onehot[:, 1], logits[:, 1])
        auc_IVH = auc(fpr_IVH, tpr_IVH)
        fpr_normal, tpr_normal, thresholds_normal = roc_curve(y_onehot[:, 2], logits[:, 2])
        auc_normal = auc(fpr_normal, tpr_normal)
        fpr_CPH, tpr_CPH, thresholds_CPH = roc_curve(y_onehot[:, 3], logits[:, 3])
        auc_CPH = auc(fpr_CPH, tpr_CPH)
        fpr_SAH, tpr_SAH, thresholds_SAH = roc_curve(y_onehot[:, 4], logits[:, 4])
        auc_SAH = auc(fpr_SAH, tpr_SAH)
        fpr_SDH, tpr_SDH, thresholds_SDH = roc_curve(y_onehot[:, 5], logits[:, 5])
        auc_SDH = auc(fpr_SDH, tpr_SDH)

        # start plot
        plt.figure()
        plt.xlim((-0.05, 1))
        plt.ylim((0., 1.05))
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.plot(fpr_EDH, tpr_EDH, color="b", lw=1, label='EDH(AUC=%0.2f)' % auc_EDH)
        plt.plot(fpr_IVH, tpr_IVH, color="g", lw=1, label='IVH(AUC=%0.2f)' % auc_IVH)
        plt.plot(fpr_normal, tpr_normal, color="r", lw=1, label='normal(AUC=%0.2f)' % auc_normal)
        plt.plot(fpr_CPH, tpr_CPH, color="c", lw=1, label='CPH(AUC=%0.2f)' % auc_CPH)
        plt.plot(fpr_SAH, tpr_SAH, color="y", lw=1, label='SAH(AUC=%0.2f)' % auc_SAH)
        plt.plot(fpr_SDH, tpr_SDH, color="k", lw=1, label='SDH(AUC=%0.2f)' % auc_SDH)
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.legend(loc="lower right")
        plt.savefig('ROC_ResNet18.png')
        print('done')

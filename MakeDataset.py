import  torch
import  os, glob
import  random, csv
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms # Used to convert an image into a pixel matrix
from    PIL import Image

# Defines a class that converts data to a frame-fixed format
class makedataset(Dataset):

    def __init__(self, root, resize, mode):
        super(makedataset, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # The dictionary used to hold the category name and category label
        # Starts to get the category name and label
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label) # Print the category name and label

        # image, label
        self.images, self.labels = self.load_csv('data.csv')

        # Scale datasets
        if mode=='train': # 80%
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        elif mode=='val': # 10%
            self.images = self.images[int(0.8*len(self.images)):int(0.9*len(self.images))]
            self.labels = self.labels[int(0.8*len(self.labels)):int(0.9*len(self.labels))]
        else: # 10%
            self.images = self.images[int(0.9*len(self.images)):]
            self.labels = self.labels[int(0.9*len(self.labels)):]

    # Defines a function that holds data paths and data labels
    def load_csv(self, filename):

        # Determine if the file exists and avoid running multiple builds
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # Read files by category
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            random.shuffle(images) # Random disturb
            # Write files
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        # Create two new lists to save the image path and label
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels) # Determines if the number of images matches the label
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path => image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15), # Random rotation makes the image more suitable for the network
            transforms.CenterCrop(self.resize), # Keep the image in the center area
            transforms.ToTensor(), # numpy => tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

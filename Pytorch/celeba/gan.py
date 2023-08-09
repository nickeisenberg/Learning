import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

root = '/Users/nickeisenberg/GitRepos/DataSets_local/' 

# training set and train data loader
trainset = torchvision.datasets.CelebA(
    root=root, split='train', download=True, transform=transform
)

train_dataloader = DataLoader(
    trainset, batch_size=64, shuffle=True
)

for d in train_dataloader:
    im = d[0][0]
    break

import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.cover_files = natsorted(sorted(glob.glob(c.TRAIN_COVER_PATH + "/*." + c.format_train)))
            self.secret_files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # val
            self.cover_files = natsorted(sorted(glob.glob(c.VAL_COVER_PATH + "/*." + c.format_val)))
            self.secret_files = natsorted(sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val)))
        self.length = max(len(self.cover_files), len(self.secret_files))

    def __getitem__(self, index):
        try:
            cover_img = Image.open(self.cover_files[index % len(self.cover_files)])
            secret_img = Image.open(self.secret_files[index % len(self.secret_files)])
            cover_img = to_rgb(cover_img)
            secret_img = to_rgb(secret_img)
            cover_item = self.transform(cover_img)
            secret_item = self.transform(secret_img)
            return cover_item, secret_item

        except Exception:
            return self.__getitem__((index + 1) % self.length)

    def __len__(self):
        return self.length


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)
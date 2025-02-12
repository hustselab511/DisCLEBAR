from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
from PIL import Image


def get_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


def get_valid_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


class BaseDataset(Dataset):
    def __init__(self, df, file_path, age_mean, age_div):
        def preprocess_df(df):
            df['zscore'] = df['boneage'].map(lambda x: (x - age_mean) / age_div)
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def get_image_path(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        file_path = f"{self.file_path}/{num}.png"
        return row, file_path

    def __len__(self):
        return len(self.df)


class RSNATrainDataset(BaseDataset):
    def __init__(self, df, file_path, age_mean, age_div, img_size):
        super().__init__(df, file_path, age_mean, age_div)
        self.img_size = img_size
        self.Trans = get_train_transform(img_size)

    def __getitem__(self, index):
        row, image_path = self.get_image_path(index)
        image = Image.open(image_path).convert('L')

        return (self.Trans(image), Tensor([row['male']])), row['zscore'], row['boneage']


class RSNAValidDataset(BaseDataset):
    def __init__(self, df, file_path, age_mean, age_div, img_size):
        super().__init__(df, file_path, age_mean, age_div)
        self.img_size = img_size
        self.Trans = get_valid_transform(img_size)

    def __getitem__(self, index):
        row, image_path = self.get_image_path(index)
        image = Image.open(image_path).convert('L')

        return (self.Trans(image), Tensor([row['male']])), row['boneage']

from torch.utils.data import Dataset
import torch
from PIL import Image

class ClrDataset(Dataset):
    def __init__(self, df, transform=None):
        self.dataframe = df
        self.t_df = df.iloc[:, :-4]
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, -3]
        time_series = self.t_df.iloc[idx, :].values
        time_series = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0)

        img_path = self.dataframe.iloc[idx, -2]  # GASF
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return time_series, image, label

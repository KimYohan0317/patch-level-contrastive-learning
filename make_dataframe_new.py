import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyts.image.gaf import GramianAngularField
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def combined_tsv(data_name):
    data_path = f'./UCRArchive_2018/{data_name}/'
    train_path = f'{data_path}{data_name}_TRAIN.tsv'
    test_path = f'{data_path}{data_name}_TEST.tsv'
    
    train_df = pd.read_csv(train_path, delimiter='\t', header=None)
    test_df = pd.read_csv(test_path, delimiter='\t', header=None)
    
    le = LabelEncoder()
    for df in [train_df, test_df]:
        df.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])
        df.rename(columns={0: 'label'}, inplace=True)
    for df in [train_df, test_df]:
        df['combined_values'] = df.iloc[:, 1:].apply(lambda row: ','.join(row.astype(str)), axis=1)
        df.fillna(0, inplace=True)
    
    file_path = f'./raw_dataset/{data_name}'
    os.makedirs(file_path, exist_ok=True)
    image_folder = f'{file_path}/image'
    os.makedirs(image_folder, exist_ok=True)
    
    def save_gaf_images(df, prefix):
        n_cols = df.shape[1] - 2 
        gasf = GramianAngularField(image_size=n_cols, method='summation')
        gadf = GramianAngularField(image_size=n_cols, method='difference')
        
        gasf_image_paths = []
        gadf_image_paths = []
        
        for i in tqdm(range(df.shape[0]), desc=f'Generating {prefix} images'):
            values = df.drop(columns=['combined_values', 'label']).iloc[i, :].values.reshape(1, -1)
            
            gasf_image = gasf.fit_transform(values)[0]
            gasf_image_path = f'{image_folder}/{data_name}_{prefix}_GASF_{i}.png'
            plt.imsave(gasf_image_path, gasf_image, cmap='rainbow')
            gasf_image_paths.append(gasf_image_path)
            
            gadf_image = gadf.fit_transform(values)[0]
            gadf_image_path = f'{image_folder}/{data_name}_{prefix}_GADF_{i}.png'
            plt.imsave(gadf_image_path, gadf_image, cmap='rainbow')
            gadf_image_paths.append(gadf_image_path)
        
        df['GASF_image'] = gasf_image_paths
        df['GADF_image'] = gadf_image_paths
    
    save_gaf_images(train_df, 'train')
    save_gaf_images(test_df, 'test')
    
    for df in [train_df, test_df]:
        label_col = df.pop('label')
        df.insert(df.columns.get_loc('combined_values') + 1, 'label', label_col)
    
    train_df.to_csv(f'{file_path}/{data_name}_train_df.csv', index=False)
    test_df.to_csv(f'{file_path}/{data_name}_test_df.csv', index=False)

# data_list = ['MixedShapesRegularTrain', 'MoteStrain', 'PigCVP', ...]
# base_dir = './UCRArchive_2018'
# data_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
for data_name in data_list:
    combined_tsv(data_name)
    print(data_name)

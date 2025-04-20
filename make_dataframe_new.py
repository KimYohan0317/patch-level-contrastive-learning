import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyts.image.gaf import GramianAngularField
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def combined_tsv(data_name):
    # 파일 경로 설정
    data_path = f'/dshome/ddualab/yohan/clr/UCRArchive_2018/{data_name}/'
    train_path = f'{data_path}{data_name}_TRAIN.tsv'
    test_path = f'{data_path}{data_name}_TEST.tsv'
    
    # TSV 파일 불러오기
    train_df = pd.read_csv(train_path, delimiter='\t', header=None)
    test_df = pd.read_csv(test_path, delimiter='\t', header=None)
    
    le = LabelEncoder()
    # 첫 번째 열을 label로 인식 및 인코딩
    for df in [train_df, test_df]:
        df.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])
        df.rename(columns={0: 'label'}, inplace=True)
    
    # 값 콤마 기준 합치기
    for df in [train_df, test_df]:
        df['combined_values'] = df.iloc[:, 1:].apply(lambda row: ','.join(row.astype(str)), axis=1)
        df.fillna(0, inplace=True)
    
    # 폴더 생성
    file_path = f'/dshome/ddualab/yohan/clr/raw_dataset/{data_name}'
    os.makedirs(file_path, exist_ok=True)
    image_folder = f'{file_path}/image'
    os.makedirs(image_folder, exist_ok=True)
    
    # 이미지 변환 및 저장 함수
    def save_gaf_images(df, prefix):
        n_cols = df.shape[1] - 2  # combined_values, label 제외
        gasf = GramianAngularField(image_size=n_cols, method='summation')
        gadf = GramianAngularField(image_size=n_cols, method='difference')
        
        gasf_image_paths = []
        gadf_image_paths = []
        
        for i in tqdm(range(df.shape[0]), desc=f'Generating {prefix} images'):
            values = df.drop(columns=['combined_values', 'label']).iloc[i, :].values.reshape(1, -1)
            
            # GASF 이미지 생성 및 저장
            gasf_image = gasf.fit_transform(values)[0]
            gasf_image_path = f'{image_folder}/{data_name}_{prefix}_GASF_{i}.png'
            plt.imsave(gasf_image_path, gasf_image, cmap='rainbow')
            gasf_image_paths.append(gasf_image_path)
            
            # GADF 이미지 생성 및 저장
            gadf_image = gadf.fit_transform(values)[0]
            gadf_image_path = f'{image_folder}/{data_name}_{prefix}_GADF_{i}.png'
            plt.imsave(gadf_image_path, gadf_image, cmap='rainbow')
            gadf_image_paths.append(gadf_image_path)
        
        df['GASF_image'] = gasf_image_paths
        df['GADF_image'] = gadf_image_paths
    
    # 이미지 변환 및 저장 호출
    save_gaf_images(train_df, 'train')
    save_gaf_images(test_df, 'test')
    
    # label 열을 combined_values 다음으로 이동
    for df in [train_df, test_df]:
        label_col = df.pop('label')
        df.insert(df.columns.get_loc('combined_values') + 1, 'label', label_col)
    
    # 최종 df 저장
    train_df.to_csv(f'{file_path}/{data_name}_train_df.csv', index=False)
    test_df.to_csv(f'{file_path}/{data_name}_test_df.csv', index=False)

# 데이터 리스트에 대해 함수 호출
# data_list = ['MixedShapesRegularTrain', 'MixedShapesSmallTrain','MoteStrain','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane','PowerCons']
# data_list = ['ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','Rock', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2','StarLightCurves', 'Strawberry', 'SwedishLeaf']
data_list = ['InsectWingbeatSound']
# base_dir = '/dshome/ddualab/yohan/clr/UCRArchive_2018'
# data_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
for data_name in data_list:
    combined_tsv(data_name)
    print(data_name)

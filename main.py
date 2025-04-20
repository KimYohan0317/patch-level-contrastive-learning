import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ClrDataset
from model import ViTTransformerModel
from loss import ContrastiveLoss
from transformers import logging as transformers_logging
import pandas as pd
import os
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from utils import calculate_metrics
import logging
from datetime import datetime

# 경고 메시지 비활성화
transformers_logging.set_verbosity_error()

random.seed(0)
torch.manual_seed(0)

def contrastive_model_train(model, trn_loader, tst_loader, device, classification_criterion, contrastive_criterion, optimizer, epochs, data_name, pooling_method, alpha):
    model.train()
    best_accuracy = 0.0
    best_model_path = f"/dshome/ddualab/yohan/clr/best_model/{data_name}_best_mymodel_{pooling_method}.pth"

    for epoch in range(epochs):
        total_contrastive_loss = 0.0
        total_classification_loss = 0.0
        total = 0
        img_correct = 0
        
        for time_series, images, label in tqdm(trn_loader):
            time_series, images, label = time_series.to(device), images.to(device), label.to(device)

            optimizer.zero_grad()
            logits, patch_embeddings, token_embeddings, _ = model(time_series, images)

            contrastive_loss = contrastive_criterion(patch_embeddings, token_embeddings)
            classification_loss = classification_criterion(logits, label)
            total_loss = (1 - alpha) * classification_loss + alpha * contrastive_loss

            total_loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            img_correct += (preds == label).sum().item()
            total += label.size(0)

            total_contrastive_loss += contrastive_loss.item()
            total_classification_loss += classification_loss.item()

        img_acc = 100. * img_correct / total
        avg_contrastive_loss = total_contrastive_loss / len(trn_loader)

        log_msg = (f"[{epoch+1}/{epochs}] Train Acc: {img_acc:.2f}% | "
                   f"Classification Loss: {total_classification_loss / len(trn_loader):.4f} | "
                   f"Contrastive Loss: {avg_contrastive_loss:.4f}")
        print(log_msg)
        logging.info(log_msg)

        accuracy = contrastive_test(model, tst_loader, device, classification_criterion, contrastive_criterion)
        print(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            msg = f"✅ Best model saved with accuracy: {best_accuracy:.4f}"
            print(msg)
            logging.info(msg)
    
    print("=" * 60)
    logging.info("=" * 60)


def contrastive_test(model, tst_loader, device, classification_criterion, contrastive_criterion, average='weighted'):
    model.eval()
    total_contrastive_loss = 0.0
    total_classification_loss = 0.0
    img_correct = 0
    total_samples = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for time_series, images, label in tqdm(tst_loader):
            time_series, images, label = time_series.to(device), images.to(device), label.to(device)

            logits, patch_embeddings, token_embeddings, _ = model(time_series, images)
            contrastive_loss = contrastive_criterion(patch_embeddings, token_embeddings)
            classification_loss = classification_criterion(logits, label)

            img_pred = torch.argmax(logits, dim=-1)
            img_correct += (img_pred == label).sum().item()
            total_samples += label.size(0)

            total_contrastive_loss += contrastive_loss.item()
            total_classification_loss += classification_loss.item()

            preds.extend(img_pred.cpu().numpy())
            true_labels.extend(label.cpu().numpy())

    accuracy = calculate_metrics(true_labels, preds, average=average)
    return accuracy

def main():
    alpha = 0.5
    batch_size = 2
    out_dim = 768
    temperature = 0.1
    img_size = 384
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    pooling_method = 'average' #attention or average
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    #오늘 날짜 지정
    today = datetime.today().strftime('%Y-%m-%d-%H:%M')
    log_dir = '/dshome/ddualab/yohan/clr/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.splitext(os.path.basename(__file__))[0] + f'_{today}_mine.log'
    logging.basicConfig(
        filename=os.path.join(log_dir, log_filename),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # 여러 데이터셋에 대한 반복
    # data_names = ['Beef', 'Car', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays']
    data_names = ['SonyAIBORobotSurface2']
    # data_names = ['ElectricDevices', 'FordA', 'FordB']
    # data_names = ['Ham', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MoteStrain', 'NonInvasiveFetalECGThorax1']
    # data_names = [ 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Phoneme', 'Plane', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2']
    # data_names = [ 'StarLightCurves', 'Strawberry', 'SyntheticControl', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'Wafer', 'Wine', 'ACSF1']
    # data_names = ['BME', 'Chinatown', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel']
    # data_names = ['FreezerRegularTrain', 'FreezerSmallTrain', 'HouseTwenty', 'MelbournePedestrian', 'SmoothSubspace', 'UMD']
    for data_name in data_names:
        logging.info(f"Starting training for dataset: {data_name}")
        print(f"Starting training for dataset: {data_name}")
        
        # 모델 초기화
        label_counts = len(pd.read_csv(f'/dshome/ddualab/yohan/clr/raw_dataset/{data_name}/{data_name}_train_df.csv').iloc[:, -3].unique())
        input_shape = len(pd.read_csv(f'/dshome/ddualab/yohan/clr/raw_dataset/{data_name}/{data_name}_train_df.csv').iloc[:, :-4].columns)
        model = nn.DataParallel(ViTTransformerModel(input_shape=input_shape, num_classes=label_counts, pooling_method=pooling_method, out_dim=out_dim, device=device)) 
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()
        contrastive_criterion = ContrastiveLoss(device, temperature)
        
        # 데이터셋 로드
        train_data_path = f'/dshome/ddualab/yohan/clr/raw_dataset/{data_name}/{data_name}_train_df.csv'
        test_data_path = f'/dshome/ddualab/yohan/clr/raw_dataset/{data_name}/{data_name}_test_df.csv'
        
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        train_dataset = ClrDataset(train_df, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        
        test_dataset = ClrDataset(test_df, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
        
        # 학습 수행
        contrastive_model_train(model, train_dataloader, test_dataloader, device, criterion, contrastive_criterion, optimizer, num_epochs, data_name, pooling_method, alpha)

if __name__ == '__main__':
    main()

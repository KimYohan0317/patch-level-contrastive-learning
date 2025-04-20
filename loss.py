import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self,device, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        
    def softXEnt(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.size(0)
        return loss
    
    def forward(self, vi, ui):
        # 임베딩 정규화
        vi = F.normalize(vi, p=2, dim=1)
        ui = F.normalize(ui, p=2, dim=1)
        
        hidden1, hidden2 = vi, ui #img, text
        batch_size = hidden1.size(0)
        
        hidden1_large = hidden1
        hidden2_large = hidden2  
        
        
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / self.temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / self.temperature
        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)
        
        return 0.5 * loss_a + 0.5 * loss_b
    
class CLIPloss(nn.Module):
    def __init__(self, device, batch_size, temperature):
        super(CLIPloss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self,vi, ui):
        image_embeddings, text_embeddings = vi, ui #img, text
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        targets.to(self.device)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / self.batch_size # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, device, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, vi, ui): # vi: 이미지 패치 집합, ui: 텍스트 패치 집합
        batch_size, num_patches, dim = vi.shape # 배치사이즈, 패치의 수, 차원(ViT)
        vi = vi / vi.norm(dim=-1, keepdim=True) # L2 정규화, keepdim=True: 입력 텐서와 같은 차원 수 유지하는 역할
        ui = ui / ui.norm(dim=-1, keepdim=True)
        logits = torch.matmul(ui, vi.transpose(-1, -2)) / self.temperature # ui와 vi의 내적, 두 벡터 간의 유사도를 나타냄
        labels = torch.arange(num_patches).expand(batch_size, num_patches).to(self.device) # 각 패치가 자신과 동일한 인덱스를 가지는 벡터와 유사해야하기 때문에 0,1,2,...와 같은 인덱스 레이블을 생성
        
        loss_i = F.cross_entropy(logits, labels)# logits과 labels의 cross entropy 계산
        loss_t = F.cross_entropy(logits.transpose(-1, -2), labels) # logits를 전치하여 다시 계산
        loss = (loss_i + loss_t) / 2 # 두 loss의 평균
        return loss

import os
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


# user-defined functions
from src.metric import my_acc_auc
from src.utils import ViolenceDataset

class Trainer:
    def __init__(self, model, criterion, optimizer, cfg, device,kfold):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.model_name = cfg.model_name
        self.optimizer = optimizer 
        self.criterion = criterion
        self.kold = kfold
        self.data_root = cfg.dataroot
        
        # for save
        self.kfold_train_loss = []
        self.kfold_train_acc = []
        self.kfold_train_f1 = []
        self.kfold_train_precision = []
        self.kfold_train_recall = []
        
        self.kfold_test_loss = []
        self.kfold_test_acc = []
        self.kfold_test_f1 = []
        self.kfold_test_precision= []
        self.kfold_test_recall =[]
        self.kfold_lr = []
        
    # pprint 
    def pprint_result(status,epoch,current_lr,average_loss,average_acc,average_f1,average_precision,average_recall):
        print(f'[INFO]| {status} | Epoch: {epoch:02d} |\
        	LR: {current_lr:.4f} | Loss: {average_loss:.4f} | ACC: {average_acc:.4f} | F1: {average_f1:.4f} | Precision: {average_precision:.4f} | Recall {average_recall:.4f}')
    
    
    def save_csv(self,fold_idx):
        df = pd.DataFrame()
        df['total_train_loss']= kfold_train_loss
        df['total_train_acc'] = kfold_train_acc
        df['total_train_f1'] = kfold_train_f1
        df['total_train_precision'] = kfold_train_precision
        df['total_train_recall'] = kfold_train_recall
        df['total_test_loss'] = kfold_test_loss
        df['total_test_acc'] = kfold_test_acc
        df['total_test_f1'] = kfold_test_f1
        df['total_test_precision'] = kfold_test_precision
        df['total_test_recall'] = kfold_test_recall
        df['total_lr'] = kfold_lr
        save_name = os.path.join(config.BASE_SAVE_DIR, f'seed({seed})_{fold_idx+1}fold.csv')
        df.to_csv(save_name, index=False)

        return None 
        
    def train_test(self):
        print("[INFO] Training start!\n")
        
        dataset = ViolenceDataset(self.data_root)
        
        # K-fold Cross Validation model evaluation
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(dataset)):
            torch.cuda.empty_cache()
            if fold_idx == 0:
                print(f'SEED: {seed}')
                print(f'The length of Train set is {len(train_index)}')
                print(f'The length of Test set is {len(test_index)}')
                
            # Data loader
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
        
            train_loader = torch.utils.data.DataLoader(dataset, 
                                                       batch_size=config.TRAIN.BATCH_SIZE,
                                                       sampler=train_subsampler)
            valid_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=config.TRAIN.BATCH_SIZE, 
                                                       sampler=test_subsampler)
            
            print(f'[INFO] {fold_idx+1}/5 FOLD')
            for epoch in range(start_epoch,end_poch+1):
                # train
                average_loss, average_acc, average_f1, average_precision, average_recall  = self.one_epoch(train_loader,test=False)                   
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                # pprint
                if epoch % config.SYSTEM.PRINT_FREQ == 0 or epoch == end_poch:
                    self.pprint_result('train',epoch,current_lr,average_loss,average_acc,average_f1,average_precision,average_recall)
                    
                # save train value  
                self.kfold_train_loss.append(average_loss)
                self.kfold_train_acc.append(average_acc)
                self.kfold_train_f1.append(average_f1)
                self.kfold_train_precision.append(average_precision)
                self.kfold_train_recall.append(average_recall)
                self.kfold_lr.append(current_lr)

                # test
                average_loss, average_acc, average_f1, average_precision, average_recall  = self.one_epoch(train_loader,test=True)                   
                # save train value  
                self.kfold_test_loss.append(average_loss)
                self.kfold_test_acc.append(average_acc)
                self.kfold_test_f1.append(average_f1)
                self.kfold_test_precision.append(average_precision)
                self.kfold_test_recall.append(average_recall)

                # pprint
                if epoch % config.SYSTEM.PRINT_FREQ == 0 or epoch ==end_poch:
                    self.pprint_result('test',epoch,current_lr,average_loss,average_acc,average_f1,average_precision,average_recall)
        
                # SAVE CHECKPOINT
                if (average_acc >=  max(self.kfold_test_acc)) and (average_loss <= min(self.kfold_test_loss)):
                    weight_save_path = os.path.join(config.BASE_WEIGHT_SAVE_DIR,f'{model_name}_seed({seed})_{fold_idx+1}fold.pth' )
                    torch.save(model, weight_save_path)
                                                           
            self.save_csv(fold_idx)
     
    def one_epoch(self, train_loader, test=False):
    
        train_total_loss = 0
        train_total_acc = 0 
        train_total_f1 = 0 
        train_total_precisioin = 0
        train_total_recall = 0

        if test:
            self.model.train()
        else:
            self.model.eval()
            with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_acc, train_f1, train_precision, train_recall = my_acc_auc(outputs, targets)

            # train_value save 
            train_total_loss += loss.item()
            train_total_acc += train_acc
            train_total_f1 += train_f1
            train_total_precisioin+=train_precision
            train_total_recall+=train_recall

        average_loss = train_total_loss/len(train_loader)
        average_acc = 100.*train_total_acc/len(train_loader)
        average_f1 = train_total_f1/len(train_loader)
        average_precision = train_total_precisioin/len(train_loader)
        average_recall = train_total_recall/len(train_loader)

        return average_loss, average_acc, average_f1, average_precision, average_recall

def main(config):
    
    date = datetime.now()
    folder_name = '{}_{}_{}_{}'.format(date.month, date.day, date.hour, date.minute)
    model_name = config.model_name
    data_root = config.DATASET.ROOT
    SAVE_DIR = os.path.join(config.BASE_SAVE_DIR, model_name,folder_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    #------------------------------# 
    # SEED  
    #------------------------------# 
    seed = config.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #------------------------------# 
    # GPU allocation 
    #------------------------------# 
    device = torch.device(f'cuda:{config.SYSTEM.GPU}' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.set_device(device)
        print ('Current cuda device ', torch.cuda.current_device()) # check
        with torch.cuda.device(f'cuda:{config.SYSTEM.GPU}'):
            torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    #------------------------------# 
    # Define model 
    #------------------------------# 
    model = Transfer_r2plus1d_18()
    criterion = nn.BCEWithLogitsLoss() # last-layer without sigmoid
    optimizer = optim.Adamax(model.parameters(), lr=config.TRAIN.BASE_LR, betas=(0.8, 0.999), eps=1e-08,)
    
    #------------------------------# 
    # K-fold Cross Validation
    #------------------------------# 
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    #------------------------------#  
    # Train & TEST 
    #------------------------------# 
    trainer = Trainer(model, criterion, optimizer, config, device, kfold)
    start = time.time()
    trainer.train_test()
    end = time.time()
    print(f"[INFO]| {model_name} END ... \nTotal duration:{end_time - start_time}")
    


if __name__ == '__main__':
    main(config)
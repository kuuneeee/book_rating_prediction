import tqdm

import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel
from ._models import rmse, RMSELoss

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from collections import defaultdict
import pickle

class NeuralCollaborativeFiltering:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx=np.array((1, ), dtype=np.long)

        self.embed_dim = args.NCF_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.NCF_MLP_DIMS
        self.dropout = args.NCF_DROPOUT

        self.model = _NeuralCollaborativeFiltering(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)

                # print('y : ', y)
                # print('targrt : ',target.float())

                loss = self.criterion(y, target.float())
                # print('loss : ',loss)


                # # k-fold validation
                # cv = KFold(n_splits=10, random_state=1, shuffle=True)
                # loss = np.mean(cross_val_score(self.model, y, target.float(), scoring='accuracy', cv=cv, n_jobs=-1))

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class WideAndDeepModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.WDN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.WDN_MLP_DIMS
        self.dropout = args.WDN_DROPOUT

        self.model = _WideAndDeepModel(self.field_dims, self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)

                loss = self.criterion(y, target.float())

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class DeepCrossNetworkModel:

    def __init__(self, args, data, idx):
        super().__init__()

        self.criterion = RMSELoss()
        # self.criterion = 
        # self.criterion = torch.nn.CrossEntropyLoss()
        
        self.data = data
        self.batch_size = args.BATCH_SIZE
        self.shuffle = True

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']        # 7

        self.embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.idx = {
                    'user2idx':self.data['user2idx'],
                    'isbn2idx':self.data['isbn2idx'],
                    'age':6,
                    'loc_city2idx':self.data['idx']['loc_city2idx'],
                    'loc_state2idx':self.data['idx']['loc_state2idx'],
                    'loc_country2idx':self.data['idx']['loc_country2idx'],
                    'publisher2idx':self.data['idx']['publisher2idx'],
                    'language2idx':self.data['idx']['language2idx'],
                    'author2idx':self.data['idx']['author2idx']
                    }

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS
        self.mlp_num_layers = args.DCN_MLP_NUM_LAYERS

        self.model = _DeepCrossNetworkModel(self.field_dims, self.embed_dim, self.num_layers, self.mlp_num_layers, 
                                            mlp_dims=self.mlp_dims, idx=self.idx, args=args, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            # for i, (fields, target) in enumerate(tk0):
            for i, data in enumerate(tk0):
                # print(len(data))
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)

                # fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)

                # ########################## classification
                # target = target-1
                # ########################## classification
                loss = self.criterion(y, target.float())

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            # print('epoch:', epoch, 'validation: rmse:', rmse_score)
            print('epoch:', epoch, 'validation: loss:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        total_loss, cnt = 0, 0
        with torch.no_grad():
            # for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
            for data in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                # fields, target = fields.to(self.device), target.to(self.device)
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                # ########################## classification
                # target = target-1
                # ########################## classification
                y = self.model(fields)

                # ########################## classification
                # loss = self.criterion(y, target.long())
                # total_loss += loss
                # cnt+=1
                # ########################## classification

                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        # mean_total_loss = total_loss/cnt

        return rmse(targets, predicts) #mean_total_loss.item()


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for data in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                    fields = fields[0].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                    fields = fields[0].to(self.device)
                y = self.model(fields)
                
                # ########################## classification
                # predict = torch.argmax(y,dim=1)
                # predicts.extend(predict.tolist())
                # ########################## classification
                
                predicts.extend(y.tolist())
        for i in range(len(predicts)):
            if predicts[i]>10:
                predicts[i]=10
            elif predicts[i]<0:
                predicts[i]=1
            else:
                predicts[i]=round(predicts[i],2)



        return predicts




    def qtrain(self):
          # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        _X = self.data["train"].drop(['rating'], axis=1)
        _y = self.data['train']['rating']
        rmse_lst = []
        para_dict = defaultdict(list)
        cnt = 0
        fold = 0
        for train_index, valid_index in tqdm.tqdm(kf.split(_X)):
            X_train, X_valid = _X.iloc[train_index], _X.iloc[valid_index]
            y_train, y_valid = _y.iloc[train_index], _y.iloc[valid_index]
            
            train_dataset = TensorDataset(torch.LongTensor(X_train.values), torch.LongTensor(y_train.values))
            valid_dataset = TensorDataset(torch.LongTensor(X_valid.values), torch.LongTensor(y_valid.values))

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

            total_rmse = []    
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                tk0 = tqdm.tqdm(train_dataloader, smoothing=0, mininterval=1.0)
                print('\ntrain...')
                for i, (fields, target) in enumerate(tk0):
                    fields, target = fields.to(self.device), target.to(self.device)
                    y = self.model(fields)

                    ########################## classification
                    target = target-1
                    ########################## classification
                    loss = self.criterion(y, target.long())

                    # loss = self.criterion(y, target.float())
                    
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    if (i + 1) % self.log_interval == 0:
                        tk0.set_postfix(loss=total_loss / self.log_interval)
                        total_loss = 0


                self.model.eval()
                targets, predicts = list(), list()
                total_loss, cnt = 0, 0
                print('\nvalidation...')
                with torch.no_grad():
                    for fields, target in tqdm.tqdm(valid_dataloader, smoothing=0, mininterval=1.0):
                        fields, target = fields.to(self.device), target.to(self.device)
                        ########################## classification
                        target = target-1
                        ########################## classification
                        y = self.model(fields)

                        ########################## classification
                        loss = self.criterion(y, target.long())
                        total_loss += loss
                        cnt+=1
                        ########################## classification

                        # targets.extend(target.tolist())
                        # predicts.extend(y.tolist())
                mean_total_loss = total_loss/cnt

        # return mean_total_loss.item() #rmse(targets, predicts)
                        
                        
                #         fields, target = fields.to(self.device), target.to(self.device)
                #         y = self.model(fields)
                #         targets.extend(target.tolist())
                #         predicts.extend(y.tolist())
                # rmse_score = rmse(targets, predicts)


                # total_rmse.append(rmse_score)
                # print('epoch:', epoch, 'validation: rmse:', rmse_score)
                print('epoch:', epoch, 'validation: rmse:', mean_total_loss.item())

            for param_tensor in self.model.state_dict():
                para_dict[param_tensor].append(self.model.state_dict()[param_tensor])
            if cnt == 0:
                with open("para_dict1.pickle","wb") as fw:
                    pickle.dump(para_dict,fw)
            cnt+=1
            rmse_lst.append(np.mean(total_rmse))

            fold+=1
            print('fold:', fold, 'validation: rmse:', np.mean(total_rmse), '\n')
            
        for name,_ in self.model.named_parameters():
            self.model.get_parameter(name).data = sum(para_dict[name])/5.
    
        with open("para_dict.pickle","wb") as fw:
            pickle.dump(para_dict,fw)
        
            
        print("평균 RMSE : ",np.mean(rmse_lst))





    
    def ptrain(self):
          # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        _X = self.data["train"].drop(['rating'], axis=1)
        _y = self.data['train']['rating']
        rmse_lst = []
        para_dict = defaultdict(list)
        cnt = 0

        total_rmse = []    
        for epoch in range(self.epochs):
            total_rmse = []
            fold = 0
            for train_index, valid_index in tqdm.tqdm(kf.split(_X)):
                total_loss = 0

                X_train, X_valid = _X.iloc[train_index], _X.iloc[valid_index]
                y_train, y_valid = _y.iloc[train_index], _y.iloc[valid_index]
                
                train_dataset = TensorDataset(torch.LongTensor(X_train.values), torch.LongTensor(y_train.values))
                valid_dataset = TensorDataset(torch.LongTensor(X_valid.values), torch.LongTensor(y_valid.values))

                self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
                self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
                    
                self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

                self.model.train()
                tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
                for i, (fields, target) in enumerate(tk0):
                    fields, target = fields.to(self.device), target.to(self.device)
                    y = self.model(fields)
                    loss = self.criterion(y, target.float())
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    if (i + 1) % self.log_interval == 0:
                        tk0.set_postfix(loss=total_loss / self.log_interval)
                        total_loss = 0

                self.model.eval()
                targets, predicts = list(), list()
                print('\nvalidation...')
                with torch.no_grad():
                    for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                        fields, target = fields.to(self.device), target.to(self.device)
                        y = self.model(fields)
                        targets.extend(target.tolist())
                        predicts.extend(y.tolist())
                rmse_score = rmse(targets, predicts)
                total_rmse.append(rmse_score)
                fold+=1
                print('fold:', fold, 'validation: rmse:', rmse_score, '\n')


            if epoch%5==0:
                for param_tensor in self.model.state_dict():
                    para_dict[param_tensor].append(self.model.state_dict()[param_tensor])
                if cnt == 0:
                    with open("para_dict1.pickle","wb") as fw:
                        pickle.dump(para_dict,fw)

            
            rmse_lst.append(np.mean(total_rmse))
            print('epoch:', epoch, 'validation: rmse:', np.mean(total_rmse), '\n')


                    
        for name,_ in self.model.named_parameters():
            self.model.get_parameter(name).data = sum(para_dict[name])/5.
    
        with open("para_dict.pickle","wb") as fw:
            pickle.dump(para_dict,fw)
        
            
        print("평균 RMSE : ",np.mean(rmse_lst))
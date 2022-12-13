def qtrain(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        _X = self.data["train"].drop(['rating'], axis=1)
        _y = self.data['train']['rating']
        rmse_lst = []
        para_dict = defaultdict(list)
        cnt = 0
        for train_index, valid_index in kf.split(_X):
            X_train, X_valid = _X.iloc[train_index], _X.iloc[valid_index]
            y_train, y_valid = _y.iloc[train_index], _y.iloc[valid_index]
            
            train_dataset = TensorDataset(torch.LongTensor(X_train.values), torch.LongTensor(y_train.values))
            valid_dataset = TensorDataset(torch.LongTensor(X_valid.values), torch.LongTensor(y_valid.values))

            self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.model = _Deep_FieldAwareFactorizationMachineModel(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

            total_rmse = []    
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
                total_rmse.append(rmse_score)
                print('epoch:', epoch, 'validation: rmse:', rmse_score)

            for param_tensor in self.model.state_dict():
                para_dict[param_tensor].append(self.model.state_dict()[param_tensor])
            if cnt == 0:
                with open("para_dict1.pickle","wb") as fw:
                    pickle.dump(para_dict,fw)
            cnt+=1
            rmse_lst.append(np.mean(total_rmse))
            
        for name,_ in self.model.named_parameters():
            self.model.get_parameter(name).data = sum(para_dict[name])/5.
    
        with open("para_dict.pickle","wb") as fw:
            pickle.dump(para_dict,fw)
        
            
        print("평균 RMSE : ",np.mean(rmse_lst))
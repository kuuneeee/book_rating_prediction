import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re
import pickle

from src.data.image_data import process_img_data

def dl_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    # field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)



    # preprocessed data
    train_data = pd.read_csv(args.PREPROCESSED_PATH + '/train_preprocessed_data_4.csv')
    test_data = pd.read_csv(args.PREPROCESSED_PATH + '/test_preprocessed_data_4.csv')
    
    with open('preprocessed/idx.pkl','rb') as f:
        idx = pickle.load(f)
    

    # field_dims_preprocessed = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)
    # field_dims_preprocessed = np.array([len(user2idx), len(isbn2idx), 6, len(idx['user_mean2idx']), len(idx['isbn_mean2idx']),
    #                                     len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32) #len(idx['language2idx'])
    # all
    field_dims_preprocessed = np.array([len(user2idx), len(isbn2idx), 6, #len(idx['user_mean2idx']), #len(idx['isbn_mean2idx']),
                            len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),# len(idx['title2idx']),
                            len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)


    img_train = process_img_data(train_data, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test_data, books, user2idx, isbn2idx, train=False)


    # print(type(train_data)) # pd

    data = {
            # 'train':train,
            # 'test':test.drop(['rating'], axis=1),
            # 'train':train_data.drop(['location_city','location_state','location_country','category_high'], axis=1),
            # 'test':test_data.drop(['location_city','location_state','location_country','category_high'], axis=1),
            'train':img_train.drop(['user_mean','isbn_mean', 'book_title'], axis=1),
            'test':img_test.drop(['user_mean','isbn_mean', 'book_title'], axis=1),
            # 'field_dims':field_dims,
            'field_dims':field_dims_preprocessed,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'test_img':test,
            'idx':idx
            }
    # print(type(data)) # dict


    return data, idx

def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

class Image_Dataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        self.user_isbn_vector = user_isbn_vector
        self.img_vector = img_vector
        self.label = label
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def dl_data_loader(args, data):
    # train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    # valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    # test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataset = Image_Dataset(
                                data['X_train'].drop(['img_vector', 'img_path'], axis=1).values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'].drop(['img_vector', 'img_path'], axis=1).values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['test'].drop(['img_vector', 'img_path'], axis=1).values,
                                data['test']['img_vector'].values,
                                # 0
                                data['test_img']['rating'].values
                                )

    # train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data

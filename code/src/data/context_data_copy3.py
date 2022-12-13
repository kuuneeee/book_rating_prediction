from operator import mod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re
import pickle
import tqdm

from src.data.image_data import process_img_data

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2):
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan)
    users = users.replace('', np.nan)
    
    modify_city = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values    

    location_list = []
    print('location processing...')
    for location in tqdm.tqdm(modify_city):
        try:
            right_city = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_city)
        except:
            pass
        
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
    
    users = users.drop(['location'], axis=1)
    
    
       
    ###########
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    ###########
    
    
    

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'book_title', 'isbn_mean', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'book_title', 'isbn_mean', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'book_title', 'isbn_mean', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # age
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mode()))   # 최빈값
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mode()))
    test_df['age'] = test_df['age'].apply(age_map)
    
    # book 파트 인덱싱
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    
    title2idx = {v:k for k,v in enumerate(context_df['book_title'].unique())}


    train_df['book_title'] = train_df['book_title'].map(title2idx)
    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)

    test_df['book_title'] = test_df['book_title'].map(title2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    user_mean2idx = {v:k for k,v in enumerate(context_df['user_mean'].unique())}
    isbn_mean2idx = {v:k for k,v in enumerate(context_df['isbn_mean'].unique())}




    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)

    train_df['user_mean'] = train_df['user_mean'].map(user_mean2idx)
    train_df['isbn_mean'] = train_df['isbn_mean'].map(isbn_mean2idx)

    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    test_df['user_mean'] = test_df['user_mean'].map(user_mean2idx)
    test_df['isbn_mean'] = test_df['isbn_mean'].map(isbn_mean2idx)


    
    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "user_mean2idx":user_mean2idx,
        "isbn_mean2idx":isbn_mean2idx,
        "title2idx":title2idx
    }

    with open('preprocessed/idx.pkl','wb') as f:
        pickle.dump(idx,f)


    return idx, train_df, test_df



def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')


    # # print()
    # print('train-test')
    # print(f'train length : {len(train)}, test length : {len(test)}')    # 306795 76699
    # # print(len(set(train['user_id']) - set(test['user_id'])))    # 41902
    # # print(len(set(test['user_id']) - set(train['user_id'])))    # 8266
    # # print(len(set(train['user_id'].unique())))                  # 59803
    # # print(len(set(test['user_id'].unique())))                   # 26467
    # print(f'user length : {len(users)}, books length : {len(books)}')   # 68092 149570

    
    # # publisher
    # publisher_dict=(books['publisher'].value_counts()).to_dict()
    # publisher_count_df = pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count']).sort_values(by=['count'], ascending = False)
    
    # modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    # print('publisher modifying...')
    # for publisher in tqdm.tqdm(modify_list):
    #     try:
    #         number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]            #####
    #         right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
                
    #         books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
    #         # print(right_publisher)
    #     except: 
    #         pass
    
    
    # # # category
    # # books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    # # books['category'] = books['category'].str.lower()
    # # category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
    # # category_df.columns = ['category','count']
    # # books['category_high'] = books['category'].copy()
    
    # # categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    # # 'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    # # 'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    # # 'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    # # for category in categories:
    # #     books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
        
    # # category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    # # category_high_df.columns = ['category','count']
    
    # # others_list = category_high_df[category_high_df['count']<5]['category'].values
    # # books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'
    
    
    # # language
    # lan = {0:'en', 1:'en', 2:'fr', 3:'de', 4:'ja', 5:'ru', 6:'it', 7:'zh-CN', 8:'es', 9:'nl', 'B':'ko'}
    # not_int = []
    # print('language processing...')
    # for i in tqdm.tqdm(range(len(books['isbn']))):
    #     idx = books['isbn'][i][0]
    #     if idx in '0123456789':
    #         books.loc[ books[books['isbn']==books['isbn'][i]].index, 'language'] = lan[int(idx)]
    #         # print(books.loc[books[books['isbn']==books['isbn'][i]].index,'language'])
    #     elif idx == 'B':
    #         books.loc[ books[books['isbn']==books['isbn'][i]].index, 'language'] = lan[idx]
    #         # print(books.loc[books[books['isbn']==books['isbn'][i]].index,'language'])



    # # 인덱싱
    print('indexing...')
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    train['isbn'] = train['isbn'].map(isbn2idx)

    test['user_id'] = test['user_id'].map(user2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    sub['user_id'] = sub['user_id'].map(user2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    
    users['user_id'] = users['user_id'].map(user2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)
    print('indexing done')


    # # m1 = np.mean(train_mean['rating'])
    # # train['user_mean'], train['isbn_mean'] = m1, m1
    # # test['user_mean'], test['isbn_mean'] = m1, m1

    # print('apply user_id...rating mean')
    # user_mean_dict = dict(train.groupby('user_id')['rating'].mean())
    # user_average = [0 for _ in range(len(train['user_id'].unique()))]
    # for i in user_mean_dict.keys():
    #     user_average[i] = round(user_mean_dict[i],2)
    # users['user_mean'] = pd.Series(user_average)


    # print('===================================================')

    # print('apply isbn...rating mean')
    # book_mean_dict = dict(train.groupby('isbn')['rating'].mean())
    # book_average = [0 for _ in range(len(train['isbn'].unique()))]
    # for i in book_mean_dict.keys():
    #     book_average[i] = round(book_mean_dict[i],1)
    # books['isbn_mean'] = pd.Series(book_average)



    # print('mean rating applyed')
    # train.to_csv('preprocessed/mean_user_isbn_train.csv', index=False, mode='w')
    # test.to_csv('preprocessed/mean_user_isbn_test.csv', index=False, mode='w')
    # print('=====================================')
    # print('preprocessed data saved')




    # idx, context_train, context_test = process_context_data(users, books, train, test)

    context_train = pd.read_csv(args.PREPROCESSED_PATH + '/train_preprocessed_data_4.csv')
    context_test = pd.read_csv(args.PREPROCESSED_PATH + '/test_preprocessed_data_4.csv')

    with open('preprocessed/idx.pkl','rb') as f:
        idx = pickle.load(f)

    
    field_dims = np.array([len(user2idx), len(isbn2idx), 6, len(idx['loc_city2idx']), len(idx['loc_state2idx']),
                        len(idx['loc_country2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']), len(idx['user_mean2idx']), 
                        len(idx['isbn_mean2idx']), len(idx['title2idx'])], dtype=np.uint32)

    img_train = process_img_data(context_train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(context_test, books, user2idx, isbn2idx, train=False)

    print('context data saved..')
    context_train.to_csv('preprocessed/context_data_train.csv', index=False, mode='w')
    context_test.to_csv('preprocessed/context_data_test.csv', index=False, mode='w')
    print('=====================================')
    print('context data saved')

    data = {
            'train': img_train, #context_train,
            'test': img_test, #context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }
    
    
    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data

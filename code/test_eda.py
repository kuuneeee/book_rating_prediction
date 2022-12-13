import pandas as pd

train_data = pd.read_csv('preprocessed/train_preprocessed_data_2.csv')
test_data = pd.read_csv('preprocessed/test_preprocessed_data_2.csv')


print('========================================================================')

train_user = list(train_data['user_id'].unique())
test_user = list(test_data['user_id'].unique())

user = train_user+test_user
print(f'train_user : {len(train_user)} test_user : {len(test_user)}')
print('안겹치는 유저 ',len(user) - len(set(user)))
# print('train에만 있는 user',len(set(user))-len(train_user))
# print('test에만 있는 user',len(set(user))-len(test_user))

print('========================================================================')

train_b = list(train_data['isbn'].unique())
test_b = list(test_data['isbn'].unique())

book = train_b+test_b

print(f'train_b : {len(train_b)} test_b : {len(test_b)}')
print('안겹치는 책 ',len(book) - len(set(book)))
# inter_train = set(train_b) & set(book)
# inter_test = set(test_b) & set(book)
# print('train에만 있는 book',len(inter_train))
# print('test에만 있는 book',len(inter_test))

print('========================================================================')

train_u_m = train_data['user_mean']
test_u_m = test_data['user_mean']

train_b_m = train_data['isbn_mean']
test_b_m = test_data['isbn_mean']

# print(train_data[train_data['user_mean']==2142].count())
# print(test_data[test_data['user_mean']==2142].count())
# print(train_data[train_data['isbn_mean']==1268].count())
# print(test_data[test_data['isbn_mean']==1268].count())


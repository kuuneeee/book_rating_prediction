import argparse
import pandas as pd

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss

from src.data import dl_data_load, dl_data_split, dl_data_loader
from src import NeuralCollaborativeFiltering, WideAndDeepModel, DeepCrossNetworkModel




class _Optimizer:
    def __init__(self, space, data):
        self.space = space
        self.data = data

    # 목적 함수 정의
    def _hyperparameter_tuning(self, space):
        class args:
            DCN_EMBED_DIM = 128
            DCN_MLP_DIMS = (1152,1152)
            DCN_NUM_LAYERS = 10 # space['DCN_NUM_LAYERS']
            DCN_MLP_NUM_LAYERS = 2 # space['DCN_MLP_NUM_LAYERS']
            DEVICE = 'cuda'
            SEED = 42
            BATCH_SIZE = 2048
            EPOCHS = 10
            DCN_DROPOUT = space['DCN_DROPOUT']
            LR = space['LR']
            # LR = 2e-3
            WEIGHT_DECAY = space['WEIGHT_DECAY']

        model = DeepCrossNetworkModel(args, self.data)
        model.train()
        rmse_v = model.predict_train()
        
        return {'loss':rmse_v, 'status': STATUS_OK, 'model': model}
    
    def find_best_param(self):
        trials = Trials()
        best = fmin(fn=self._hyperparameter_tuning,
                  space=self.space,
                  algo=tpe.suggest,
                  max_evals=50, 
                  trials=trials,
                  early_stop_fn=no_progress_loss(10))
        return best



def main(args):
    # hp.choice(하이퍼파라미터 이름, 후보군 리스트): 후보군 리스트 중에서 하나를 선택하여 대입하면서 최적의 하이퍼 파라미터를 찾습니다.
    # hp.quniform(하이퍼 파라미터 이름, start, end, step): start, end까지 step 간격으로 생성된 후보군 중에서 최적의 하이퍼 파라미터를 찾습니다.
    # hp.uniform(하이퍼 파라미터 이름, start, end): start, end 사이의 임의의 값 중에서 최적의 하이퍼 파라미터를 찾습니다.
    space = {}
    space['LR'] = hp.uniform('LR', 0.0001, 0.01)
    space['DCN_DROPOUT'] = hp.uniform('WDN_DROPOUT', 0.01, 0.2)
    space['WEIGHT_DECAY'] = hp.uniform('WEIGHT_DECAY', 1e-7, 1e-5)

    # space['BATCH_SIZE'] = hp.choice('BATCH_SIZE', [i for i in range(64, 1024, 64)])
    # space['DCN_NUM_LAYERS'] = hp.choice('DCN_NUM_LAYERS', [i for i in range(1, 10, 1)])
    # space['DCN_MLP_NUM_LAYERS'] = hp.choice('DCN_MLP_NUM_LAYERS', [i for i in range(1, 10, 1)])


    data = dl_data_load(args)
    data = dl_data_split(args, data)
    data = dl_data_loader(args, data)

    opt = _Optimizer(space, data)
    opt.find_best_param()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg('--DATA_PATH', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--PREPROCESSED_PATH', type=str, default='preprocessed/', help='preprocessed data')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    
    arg('--BATCH_SIZE', type=int, default=512, help='Batch size를 조정할 수 있습니다.')    # 1024
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')            # 10
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')       # 1e-3
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    args = parser.parse_args()
    main(args)
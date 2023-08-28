import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation
from utils import check_convention, config_path_prefix

DEFAULT_RES_SAVE_PATH_PREFIX = "./res/"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
    eval_res = trainer.final_eval(test_loader)
    
    start_eval_time = time.time()
    pred_all, y_all = trainer.test(all_loader)
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time) 
    recorder.record_time(eval_time)
    pred_matrix = dataloader.reconstruct_pred(pred_all)


    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)

    return recorder

def train_convention_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    trainX, trainY, testX, testY, allX = dataloader.generate_torch_dataset() 
    print('99999')

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)
    print('aaaaa')

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)
    print('bbbbb')

    return recorder 




include_path = [
    # 'indian_diffusion.json',
    'pavia_diffusion.json',
    # 'salinas_diffusion.json',
]
def run_one(param):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    name = param['net']['trainer']
    convention = check_convention(name)
    uniq_name = param.get('uniq_name', name)
    print('start to train %s...' % uniq_name)
    if convention:
        train_convention_by_param(param)
    else:
        train_by_param(param)
    print('model eval done of %s...' % uniq_name)
    path = '%s/%s' % (save_path_prefix, uniq_name) 
    recorder.to_file(path)


def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        convention = check_convention(name)
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        print('start to train %s...' % uniq_name)
        if convention:
            train_convention_by_param(param)
        else:
            train_by_param(param)
        print('model eval done of %s...' % uniq_name)
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        recorder.to_file(path)


def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False

def run_diffusions():
    sample_num = 30
    config_name = 'salinas_diffusion.json'
    tlist = [5, 10, 50, 100, 200]
    layers = [0, 1, 2]

    path_param = '%s/%s' % (config_path_prefix, config_name)
    with open(path_param, 'r') as fin:
        params = json.loads(fin.read())
        for t in tlist:
            for l in layers:
                res_file_part = "%s_%s_%s" %(config_name, t, l) 
                if result_file_exists(DEFAULT_RES_SAVE_PATH_PREFIX, res_file_part):
                    print(res_file_part, "exits, now continue...")
                    continue
                data_sign = params['data']['data_sign']
                uniq_name = "%s_%s_%s" % (config_name, t, l)
                params['uniq_name'] = uniq_name
                params['data']['data_file'] = '%s_%s' % (data_sign, sample_num)
                params['data']['diffusion_data_sign'] = 't%s_%s_full.pkl.npy' % (t, l)
                print("schedule %s..." % uniq_name)
                # subprocess.run('python ./workflow.py', shell=True)
                run_one(params)
                print("schedule done of %s..." % uniq_name)


if __name__ == "__main__":
    run_all()
    
    





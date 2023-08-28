import os
import json
from utils import check_convention, config_path_prefix
from workflow import train_by_param, train_convention_by_param
import subprocess
import workflow

exchange_json_file='%s/temp.json' % config_path_prefix

def simple_run_times():
    times = 1 #每个配置跑5次
    sample_num = [80]
    #times = 1 #每个配置跑5次
    #sample_num = [40]

    configs = [
        #'indian_cross_param_use.json',
        #'pavia_cross_param_use.json',
        #'salinas_cross_param_use.json',

        #'indian_diffusion.json',
        # 'pavia_diffusion.json',
        'salinas_diffusion.json',

        #'indian_ssftt.json',
        #'pavia_ssftt.json',
        #'salinas_ssftt.json',

       # 'indian_conv1d.json',
       # 'pavia_conv1d.json',
       # 'salinas_conv1d.json',

       # 'indian_conv2d.json',
       # 'pavia_conv2d.json',
       # 'salinas_conv2d.json',
    ]
    for config_name in configs:
        path_param = '%s/%s' % (config_path_prefix, config_name )
        with open(path_param, 'r') as fin:
            params = json.loads(fin.read())
            data_sign = params['data']['data_sign']
            for num in sample_num:
                for t in range(times):
                    uniq_name = "%s_%s_%s" % (config_name, num, t)
                    params['data']['data_file'] = '%s_%s' % (data_sign, num)
                    params['uniq_name'] = uniq_name
                    with open(exchange_json_file,'w') as fout:
                        json.dump(params,fout)
                    print("schedule %s..." % uniq_name)
                    # subprocess.run('python ./workflow.py', shell=True)
                    workflow.run_all()
                    print("schedule done of %s..." % uniq_name)


                
def simple_run_diffusion_t_layer():
    times = 1 #每个配置跑5次
    sample_num = 30
    config_name = 'indian_diffusion.json'
    tlist = [5, 10, 50, 100, 200]
    layers [0, 1, 2]

    path_param = '%s/%s' % (config_path_prefix, config_name)
    with open(path_param, 'r') as fin:
        params = json.loads(fin.read())
        for t in tlist:
            for l in layers:
                data_sign = params['data']['data_sign']
                uniq_name = "%s_%s_%s" % (config_name, t, l)
                params['uniq_name'] = uniq_name
                params['data']['diffusion_data_sign'] = 't%s_%s_full.pkl.npy' % (data_sign, num)
                with open(exchange_json_file,'w') as fout:
                    json.dump(params,fout)
                print("schedule %s..." % uniq_name)
                # subprocess.run('python ./workflow.py', shell=True)
                workflow.run_all()
                print("schedule done of %s..." % uniq_name)

if __name__ == "__main__":
    simple_run_times()

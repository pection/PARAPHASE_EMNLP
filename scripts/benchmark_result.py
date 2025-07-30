import os 
import glob
import argparse
import pandas as pd
import numpy as np
def std(x):
    return np.std(x)
def collect_results(args):
    all_experiment_dirs = glob.glob(f'{args.exp_root}/*/*')
    print('These are all the experiment directories')
    print(all_experiment_dirs)
    
    results = []
    for exp_dir in all_experiment_dirs:
        # if 't5-3b' in exp_dir and 't5-base' in exp_dir:
        #     print('Skipping this directory', str(exp_dir), ' because it has t5-3b or t5-base in it')
        #     continue
        if not os.path.isdir(exp_dir):
            continue

        log_path = os.path.join(exp_dir, 'logger.log') #path to log file
        metrics = ['dev_acc', 'dev_bertscore_correct_normalized'] #metrics being monitored
        metrics_dict = {} #stores the monitored metrics for each experiment
        #check log file for the current experiment, and extract the monitored metrics
        with open(log_path) as f:
            for line in f:
                for metric in metrics: 
                    if (metric in line and f"{metric}_correct_pred" not in line) or (metric in line and metric.replace("_correct_pred","") not in line):
                        score = line.strip().split()[-1]
                        metrics_dict[metric] = float(score)

        config_path = os.path.join(exp_dir, 'commandline_args.txt') #path to file storing experiment configurations
        #extract configs (to be used in dataframe) from config file
        with open(config_path) as f:
            lines = f.readlines()
            lines = [l for l in lines[5:]]
            configs = ''.join(lines).replace('\n', '').split('--')
            filtered_configs = {}
            for metric in metrics: 
                filtered_configs[metric] = -1 
            configs_to_keep = ['task_name', 'model_type', 'n_shots', 'seed', 'io_format'] 
            for config in configs:
                for config_name in configs_to_keep:
                    if config.startswith(config_name):
                        config = config.replace(config_name, '')
                        config = config.replace('../checkpoints/', '')  
                        filtered_configs[config_name] = config
                        break
        filtered_configs.update(metrics_dict)
        if filtered_configs['dev_acc'] == -1:
            print (f"There is a failed run. Path to its log file: {log_path}.\n \
                   Remove this serailization directiory from `all_experiment_dirs` if not needed or\
                   re-run this experiment.")
            print(filtered_configs.values())
        else:
            results.append(filtered_configs)

    print('The number of experiments that have metrics calculated is', str(len(results)), ' out of ', str(len(all_experiment_dirs)), 'experiments.')
    df = pd.DataFrame.from_records(results) 
    try:
        assert len(df) % len(seeds_fewshot) == 0 
    except AssertionError: 
        for model in set(df['model_type'].tolist()):
            df_sub = df[df['model_type']==model]
            for io_format in set(df_sub['io_format'].tolist()):
                if len (df_sub[df_sub['io_format']==io_format]) != len(seeds_fewshot):
                    seeds = [x for x in seeds_fewshot if str(x) not in df_sub[df_sub['io_format']==io_format]["seed"].tolist()]
                    print(f"The following seeds failed for model {model} and format {io_format}:")
                    print (seeds)
                    print (f"Repeat experiments for those seeds and collect results again")
    
    #make results directory
    output_file = args.output
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    columns_to_convert = ['seed', 'n_shots']

    df[columns_to_convert] = df[columns_to_convert].astype('int64')
    result_all_path =os.path.join(output_file,"result_all.csv")
    print(df.to_csv(result_all_path, index=True))
    result_path =os.path.join(output_file,"result.csv")

    df_avg_seed = df.groupby(['task_name', 'model_type', 'io_format', 'n_shots']).mean() #key error would occur when the results that was converted to df was empty
    print(df_avg_seed.to_csv(result_path, index=True))
    result_std_path =os.path.join(output_file,"result_with_std.csv")

    df_avg_seed_with_std = df.groupby(['task_name', 'model_type', 'io_format', 'n_shots']).agg(['mean', std]) 
    print(df_avg_seed_with_std.to_csv(result_std_path, index=True))

seeds_fewshot = [7004, 3639, 6290, 9428, 7056, 4864, 4273, 7632, 2689, 8219, 4523, 2175, 7356, 8975, 51, 4199, 4182, 1331, 2796, 6341, 7009, 1111, 1967, 1319, 741, 7740, 1335, 9991, 6924, 6595, 5358, 2638, 6227, 8384, 2769, 9933, 6339, 3112, 1349, 8483, 2348, 834, 6895, 4823, 2913, 9962, 178, 2147, 8160, 1936, 4512, 2051, 4779, 2498, 176, 9599, 1181, 5320, 588, 4791]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, help="path to a directory where checkpoints will be saved")
    parser.add_argument("--output", type=str, help="path to a directory where checkpoints will be saved")

    args = parser.parse_args()
    collect_results(args)
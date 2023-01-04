import time, os, argparse,yaml
from inference import Inference

def write_config(config, outfile):
    with open(outfile, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def get_current_time():
    now = int(round(time.time()*1000))
    current_time = time.strftime('%H%M%S', time.localtime(now/1000))
    current_day = time.strftime('%Y_%m_%d', time.localtime(now/1000))
    return current_time, current_day

def mkdir_output_test(args):
    GPU_ID = args['GPU_ID']
    dir_output = args['output']['dir_output']

    current_time, current_day = get_current_time()

    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    dir_output = os.path.join(dir_output, current_day, str(gpu_id)+'_'+current_time)

    dir_img_result = os.path.join(dir_output, 'image')
    dir_img_video = os.path.join(dir_output, 'video')
    dir_config = os.path.join(dir_output, 'config')

    os.makedirs(dir_img_result, exist_ok=True)
    os.makedirs(dir_img_video, exist_ok=True)
    os.makedirs(dir_config, exist_ok=True)

    write_config(args, os.path.join(dir_config,'testing_config.yaml'))

    return dir_img_result, dir_img_video


parses = argparse.ArgumentParser()
parses.add_argument('--config', type=str, default='./config/testing.yaml', help='path to the configs file.')
opts = parses.parse_args()

def get_config(filename):
    with open(r''+filename) as file:
        list = yaml.load(file, Loader=yaml.FullLoader)
        return list
def main():
    args = get_config(opts.config)
    if args is None:
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args['GPU_ID']))
    dir_result = mkdir_output_test(args)
    inference = Inference(args)
    inference.predict_test(dir_result)

if __name__ == '__main__':
    main()
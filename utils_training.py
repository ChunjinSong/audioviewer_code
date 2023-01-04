import argparse
import yaml, time, os

# def get_config(config):
#     with open(config,'r') as stream:
#         return yaml.load(stream)

def get_config(filename):
    with open(r''+filename) as file:
        list = yaml.load(file, Loader=yaml.FullLoader)
        return list

def get_current_time():
    now = int(round(time.time()*1000))
    current_time = time.strftime('%H%M%S', time.localtime(now/1000))
    current_day = time.strftime('%Y_%m_%d', time.localtime(now/1000))
    return current_time, current_day

def write_config(config, outfile):
    with open(outfile, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def mkdir_output_train(args):
    GPU_ID = args['trainer']['GPU_ID']
    dir_output = os.path.join(args['output']['dir_output'])

    current_time, current_day = get_current_time()

    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    dir_output = os.path.join(dir_output, current_day, str(gpu_id)+'_'+current_time)

    dir_log = os.path.join(dir_output, 'log')
    dir_model = os.path.join(dir_output, 'model')
    dir_img_result = os.path.join(dir_output, 'image')
    dir_config = os.path.join(dir_output, 'config')

    os.makedirs(dir_log, exist_ok=True)
    os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_img_result, exist_ok=True)
    os.makedirs(dir_config, exist_ok=True)

    write_config(args, os.path.join(dir_config,'test_config.yaml'))

    return dir_log, dir_model, dir_img_result


def mkdir_output_test(args):
    GPU_ID = args['GPU_ID']
    dir_output = args['output']['dir_output']

    current_time, current_day = get_current_time()

    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    dir_output = os.path.join(dir_output, current_day, str(gpu_id)+'_'+current_time)

    # dir_log = os.path.join(dir_output, 'log')
    # dir_model = os.path.join(dir_output, 'model')
    dir_img_result = os.path.join(dir_output, 'image')
    dir_img_video = os.path.join(dir_output, 'video')
    dir_config = os.path.join(dir_output, 'config')

    dir_img_result_q = os.path.join(dir_output, 'image_questionnaire')
    dir_img_video_q = os.path.join(dir_output, 'video_questionnaire')

    # os.makedirs(dir_log, exist_ok=True)
    # os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_img_result, exist_ok=True)
    os.makedirs(dir_img_video, exist_ok=True)
    os.makedirs(dir_config, exist_ok=True)
    os.makedirs(dir_img_result_q, exist_ok=True)
    os.makedirs(dir_img_video_q, exist_ok=True)

    write_config(args, os.path.join(dir_config,'test_config.yaml'))

    return dir_img_result, dir_img_video, dir_img_result_q, dir_img_video_q, dir_output

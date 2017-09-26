import subprocess
import re
from tensorboard_logging import Logger
import argparse
import os
import signal

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',help='log dir')
parser.add_argument('--mode', default='all', help='logging mode, user or all, default: [all]')
parser.add_argument('--cmdline',default='echo "please give a cmdline"', help='cmdline to run caffe train')
parser.add_argument('--logging_groups', default='train,test', help='comma separated list of logging groups, default: [train,test]')
args = parser.parse_args()

if args.mode == 'all':
    args.logging_groups = 'train,test'

LOG_DIR = args.log_dir
loggers = []
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
logging_groups = args.logging_groups.split(',')
for logging_group in logging_groups:
    logging_group = LOG_DIR+'/'+logging_group
    if not os.path.exists(logging_group): os.mkdir(logging_group)
    loggers.append(Logger(logging_group))


dev_null = open('/dev/null','w')

def data_to_log(logging_type, logging_tag, logging_group, regex, regex_group_id):
    return dict(logging_type=logging_type, logging_tag=logging_tag, logging_group=logging_group, regex=regex, regex_group_id=regex_group_id)

patterns_user = [
           data_to_log('step', 'step', 0, 'Iteration (\d+)', 1),
           data_to_log('scalar', 'acc_1', 0, 'Train net output #0: Accuracy1 = (\d+\.\d+)', 1),
           data_to_log('scalar', 'loss_cls_1', 0, 'Train net output #1: SoftmaxWithLoss1 = (\d+\.\d+)', 1)]
patterns_for_all = [
           data_to_log('step', 'step', None, 'Iteration (\d+)', 1),
           data_to_log('scalar', None, 0, 'Train net output #(\d+): (\w*) = (\d+\.*\d+)', None),
           data_to_log('scalar', None, 1, 'Test net output #(\d+): (\w*) = (\d+\.*\d+)', None)]
# cmd = '$PROOT/caffe/build/tools/caffe train -gpu 6 -solver solver.prototxt'

def parser_user(line, patterns=patterns_user):
    value = None
    for pattern in patterns:
        #from IPython import embed; embed()
        logging_type = pattern['logging_type']
        logging_tag = pattern['logging_tag']
        logging_group = pattern['logging_group']
        regex = pattern['regex']
        regex_group_id = pattern['regex_group_id']
        result = re.search(regex, line)
        if result: # and len(result.groups)>group_id:
            if logging_type == 'scalar':
                #from IPython import embed; embed()
                value = float(result.group(regex_group_id))
            if logging_type == 'step':
                #from IPython import embed; embed()
                value = int(result.group(regex_group_id))
            break
    return logging_type, logging_tag, logging_group, value

def parser_all(line, patterns=patterns_for_all):
    value = None
    for pattern in patterns:
        #from IPython import embed; embed()
        logging_type = pattern['logging_type']
        logging_tag = pattern['logging_tag']
        logging_group = pattern['logging_group']
        regex = pattern['regex']
        regex_group_id = pattern['regex_group_id']
        result = re.search(regex, line)
        if result: # and len(result.groups)>group_id:
            if logging_type == 'scalar':
                #from IPython import embed; embed()
                logging_tag = result.group(2)
                value = float(result.group(3))
            if logging_type == 'step':
                #from IPython import embed; embed()
                value = int(result.group(regex_group_id))
            break
    return logging_type, logging_tag, logging_group, value

proc = subprocess.Popen(args.cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

global_step = 0
parser = parser_user if args.mode=='user' else parser_all
while True:
  try:
    line = proc.stdout.readline()
  except Exception as e:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups
  if line != '':
    #the real code does filtering here
    print "stdout:", line.rstrip()
    logging_type, logging_tag, logging_group, value = parser(line.rstrip())
    if value is None:
        pass
    elif logging_type == 'scalar':
        loggers[logging_group].log_scalar(logging_tag, value, global_step)
        print logging_tag, value
    elif logging_type == 'step':
        global_step = value
        print logging_tag, value
    else:
        pass
  else:
    break


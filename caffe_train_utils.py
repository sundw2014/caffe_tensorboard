import subprocess
import re
from tensorboard_logging import Logger

dev_null = open('/dev/null','w')

def data_to_log(logging_type, logging_tag, regex, group_id):
    return dict(logging_type = logging_type, logging_tag=logging_tag, regex=regex, group_id=group_id)

patterns = [data_to_log('step', 'step', 'Iteration (\d+)', 1),
           data_to_log('scalar', 'acc_1', 'Train net output #0: Accuracy1 = (\d+\.\d+)', 1),
           data_to_log('scalar', 'loss_cls_1', 'Train net output #1: SoftmaxWithLoss1 = (\d+\.\d+)', 1)]

cmd = '$PROOT/caffe/build/tools/caffe train -gpu 6 -solver solver.prototxt'

def parser(line):
    logging_type = None
    tag = None
    value = None
    for pattern in patterns:
        #from IPython import embed; embed()
        logging_type = pattern['logging_type']
        logging_tag = pattern['logging_tag']
        regex = pattern['regex']
        group_id = pattern['group_id']
        result = re.search(regex, line)
        if result: # and len(result.groups)>group_id:
            if logging_type == 'scalar':
                #from IPython import embed; embed()
                value = float(result.group(group_id))
            if logging_type == 'step':
                #from IPython import embed; embed()
                value = int(result.group(group_id))
            break
    return logging_type, logging_tag, value

logger = Logger('log')

proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
global_step = 0
while True:
  line = proc.stdout.readline()
  if line != '':
    #the real code does filtering here
    print "stdout:", line.rstrip()
    logging_type, tag, value = parser(line.rstrip())
    if value is None:
        pass
    elif logging_type == 'scalar':
        logger.log_scalar(tag, value, global_step)
        print tag, value
    elif logging_type == 'step':
        global_step = value
        print tag, value
    else:
        pass
  else:
    break


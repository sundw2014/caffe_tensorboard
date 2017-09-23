import subprocess

cmd = '$PROOT/caffe train -gpu 0 -solver solver.txt'
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

while True:
  line = proc.stdout.readline()
  if line != '':
    #the real code does filtering here
    print "test:", line.rstrip()
    # logging
  else:
    break

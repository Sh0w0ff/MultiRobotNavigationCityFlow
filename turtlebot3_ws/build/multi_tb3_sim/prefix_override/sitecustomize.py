import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sh0w0ff/turtlebot3_ws/install/multi_tb3_sim'

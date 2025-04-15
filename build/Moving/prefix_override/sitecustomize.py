import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ezy21kl/Documents/Anti-Intruder-Bot/install/Moving'

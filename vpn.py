__author__ = 'jmh081701'
import sys
import os
import time
cmd_test="ifconfig | grep ppp0"
cmd="/etc/init.d/xl2tpd start && echo 'c testvpn' > /var/run/xl2tpd/l2tp-control"
while True:
    time.sleep(60)
    out=os.popen(cmd_test).readlines()
    print(out)
    if len(out)>4:
        continue
    out=os.popen(cmd)
    out.readlines()
    print(out)
    time.sleep(2)
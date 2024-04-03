#!/bin/python3
import os
zipfiles = os.listdir()
zipfiles = list(filter(lambda x: x.endswith(".zip"), zipfiles))
for zipfile in zipfiles:
    cmd = "unzip %s" % (zipfile)
    print(cmd)
    os.system(cmd)


import lzma
import glob
import os

files = sorted(glob.glob('../logs/cut/*.xz'))
for i, file in enumerate(files):
    new_file = file.replace('t25d5_', '').replace('t40d4_', '')
    os.system('mv {} {}'.format(file, new_file))

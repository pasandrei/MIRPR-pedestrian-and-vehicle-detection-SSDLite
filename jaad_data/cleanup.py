import os
import shutil

rootdir = 'C:\\Users\Andrei Popovici\Desktop\JAAD_stuff\JAAD-JAAD_2.0\images'

for subdir, dirs, files in os.walk(rootdir):
    try:
        subdir.index("inference")
        shutil.rmtree(subdir)
    except Exception:
        pass

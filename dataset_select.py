import os
import random
import shutil
import skimage.io as io

def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)
    for filename in pathDir:
        print(filename)

    # coll = io.ImageCollection(str)
    # print(len(coll))
    # num = int((2*len(coll)))
    # print(num)
    num = 1000
    sample = random.sample(pathDir, num)

    for name in sample:
        shutil.copyfile(fileDir+name, tarDir+name)

if __name__ == '__main__':
    fileDir = '../../Dataset/ILSVRC2012_img_val/'
    tarDir = '../../Dataset/QuanDataset/'
    str = 'fileDir*+.jpg'
    copyFile(fileDir, tarDir)
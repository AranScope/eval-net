from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import skimage.io as io
import time
from main import get_label_confidence

dataDir='datasets'
dataType='val2017'
annFile='{}/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['chair']);
imgIds = coco.getImgIds(catIds=catIds);
imgs = coco.loadImgs(imgIds)

f = open('output.csv', 'w')
f.write('image_id,confidence\n')

for i, img in enumerate(imgs):
    image = io.imread(img['coco_url'])
    conf = get_label_confidence(image, 'chair')
    f.write('{},{}\n'.format(i, conf))
    print('the confidence is', (conf*100), '%')
    # if(conf == 0):
    plt.imsave('test.png', image)
    time.sleep(0.2)

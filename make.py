import glob
from PIL import Image

imgs = glob.glob("in/*")

full_size = (2048, 1536)
thumb_size = (256, 192)

img_path = "images/fulls/"
thumb_path = "images/thumbs/"

i = 1
for img in imgs:
    print img
    im = Image.open(img)
    im = im.resize(full_size)
    im.save("%s%02d.jpg" %(img_path, i), "JPEG")
    im.thumbnail(thumb_size)
    im.save("%s%02d.jpg" %(thumb_path, i), "JPEG")
    i += 1


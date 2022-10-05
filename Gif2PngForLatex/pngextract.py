# found this code on https://stackoverflow.com/questions/10269099/pil-convert-gif-frames-to-jpg

from PIL import Image
from PIL import GifImagePlugin


def gif2png(file_name: str, num_key_frames: int, trans_color: tuple):
    """
    convert gif to `num_key_frames` images with jpg format
    :param file_name: gif file name
    :param num_key_frames: result images number
    :param trans_color: set converted transparent color in jpg image
    :return:
    """
    with Image.open(file_name) as im:
        for i in range(num_key_frames):
            im.seek(im.n_frames // num_key_frames * i)
            image = im.convert("RGBA")
            datas = image.getdata()
            newData = []
            for item in datas:
                if item[3] == 0:  # if transparent
                    newData.append(trans_color)  # set transparent color in jpg
                else:
                    newData.append(tuple(item[:3]))
            image = Image.new("RGB", im.size)
            image.getdata()
            image.putdata(newData)
            image.save('rwalk2bm{}.png'.format(i))


gif2png("rwalk2bm1.gif", 19, (255, 255, 255))  # convert image.gif to 8 jpg images with white background



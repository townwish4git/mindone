import os
import sys
from PIL import Image
import numpy as np


def concat_images(images: list):
    images = [Image.open(img) for img in images]
    row = col = int(np.sqrt(len(images)))
    w = max([x.width for x in images])
    h = max([x.height for x in images])
    target_w, target_h = int(w * row), int(h * col)

    target = Image.new('RGB', (target_w, target_h), color=(255, 255, 255))
    for i in range(row):
        for j in range(col):
            target.paste(images[col*i+j], (0 + w * j, 0 + h * i))

    return target


def main():
    directory = sys.argv[1]
    n = int(sys.argv[2])
    save_path = sys.argv[3]
    
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    assert len(image_files) % n == 0
    num_list = len(image_files) // n

    concated_pic_list = []
    for i in range(num_list):
        batch = image_files[i*n: (i+1)*n]
        concated_pic = concat_images([os.path.join(directory, img) for img in batch])
        concated_pic.save(os.path.join(save_path, f"{i}.png"))


if __name__ == "__main__":
    main() 

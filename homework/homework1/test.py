name = 'a.b.pig'
print(len(name.split('.')))
print(name.split('.')[2])

import PIL as pil_image

img = pil_image.open(path)
img.resize(target_size, pil_image.NEAREST)
    
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from PIL import Image
# The generators use the same arguments as the CLI, only as parameters
generator_A = GeneratorFromStrings(
    strings=['AAAaaa'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/FloodStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/UmbraStd.ttf'],
    count =3,
    size=105,
    
)
generator_B = GeneratorFromStrings(
    strings=['BBBbbb'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/FloodStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/UmbraStd.ttf'],
    count =3,
    size=105,
)
generator_H = GeneratorFromStrings(
    strings=['HHHhhh'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/FloodStd.ttf',  '/home/yangbo/projects/font_recog/generator_tc/font_source/UmbraStd.ttf'],
    count =3,
    size=105,
)
for generator in [generator_A, generator_B, generator_H]:
    for img, lbl,font in generator:
        # Do something with the pillow images here.
        font = font.split('/')[-1].split('.')[0]
        file_name = lbl + '_' + font + '.png'
        img.save('./out/'+file_name)


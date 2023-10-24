from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from PIL import Image
# The generators use the same arguments as the CLI, only as parameters
generator_A = GeneratorFromStrings(
    strings=['Hospital'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf'],
    count =2,
    size=105,
    background_type=1
    
)
generator_B = GeneratorFromStrings(
    strings=['EARTHQUAKE'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf'],
    count =2,
    size=105,
    background_type=1,
)
generator_H = GeneratorFromStrings(
    strings=['exploration'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/CourierStd.ttf'],
    count =2,
    size=105,
    background_type=1
)
for generator in [generator_A, generator_B, generator_H]:
    for img, lbl,font in generator:
        # Do something with the pillow images here.
        font = font.split('/')[-1].split('.')[0]
        file_name = lbl + '_' + font + '.png'
        img.save('./out/'+file_name)


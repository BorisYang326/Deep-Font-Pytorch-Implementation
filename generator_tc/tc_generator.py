from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
from PIL import Image
# The generators use the same arguments as the CLI, only as parameters
# generator_A = GeneratorFromStrings(
#     strings=['Adviserss'],
#     fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/Copal-Std-Outline.ttf'],
#     count =2,
#     size=105,
#     background_type=1,
#     margins = (0,0,0,0)
    
# )
# generator_B = GeneratorFromStrings(
#     strings=['Equally'],
#     fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/Copal-Std-Outline.ttf'],
#     count =2,
#     size=105,
#     background_type=1,
# )
generator_H = GeneratorFromStrings(
    strings=['Adviser'],
    fonts=['/home/yangbo/projects/font_recog/generator_tc/font_source/Copal-Std-Outline.ttf'],
    count =2,
    size=105,
    random_skew=True,
    margins=(0,0,0,0),
)
# for generator in [generator_A, generator_B, generator_H]:
for i,(img, lbl,font) in enumerate(generator_H):
    # Do something with the pillow images here.
    font = font.split('/')[-1].split('.')[0]
    file_name = font + '_' + lbl +'_' + str(i) + '.png'
    img.save('./out/'+file_name)


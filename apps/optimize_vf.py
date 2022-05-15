import pydiffvg
import torch
import skimage
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from IPython.display import Image, display

def display_image(p):
    display(Image(p))
    
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 510, 510

target_font = TTFont("/content/diffvg/font_data/static/OpenSans-Bold.ttf")
target_glyph_set = target_font.getGlyphSet()
target_svgpen = SVGPathPen(target_glyph_set)
glyph = target_glyph_set["A"]
glyph.draw(target_svgpen)  
target_path = target_svgpen.getCommands(). replace(' ', ',')

# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path(target_path)
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/optimize_vf/target.png', gamma=2.2)
target = img.clone()

display_image('results/optimize_vf/target.png')

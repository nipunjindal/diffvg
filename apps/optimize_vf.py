import pydiffvg
import torch
import skimage
from fontTools.misc import transform
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
    
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 510, 510

target_font = TTFont("/content/diffvg/font_data/static/OpenSans-Bold.ttf")

# transformation
units_per_em = target_font['head'].unitsPerEm
ttf2em = transform.Identity.scale(1/units_per_em, 1/units_per_em)
svg_per_em=100
xform = transform.Identity.translate(0, svg_per_em).scale(svg_per_em, -svg_per_em).transform(ttf2em)

# glyph to svg
target_glyph_set = target_font.getGlyphSet()
target_svgpen = SVGPathPen(target_glyph_set)
target_svgpen_transformed = TransformPen(target_svgpen, xform)
glyph = target_glyph_set["C"]
glyph.draw(target_svgpen_transformed)  
target_path = target_svgpen.getCommands()

# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path(target_path)
# print(target_path)
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

print("display_image('results/optimize_vf/target.png')")

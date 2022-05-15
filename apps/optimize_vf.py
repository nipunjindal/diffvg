import pydiffvg
import torch
import skimage
from fontTools.misc import transform
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.varLib import instancer, mutator

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 510, 510

target_font = TTFont("/content/diffvg/font_data/static/OpenSans-Bold.ttf")

# transformation
target_units_per_em = target_font['head'].unitsPerEm
target_ttf2em = transform.Identity.scale(1/target_units_per_em, 1/target_units_per_em)
target_svg_per_em=100
target_xform = transform.Identity.translate(0, target_svg_per_em).scale(target_svg_per_em, -target_svg_per_em).transform(target_ttf2em)

# glyph to svg
target_glyph_set = target_font.getGlyphSet()
target_svgpen = SVGPathPen(target_glyph_set)
target_svgpen_transformed = TransformPen(target_svgpen, target_xform)
target_glyph = target_glyph_set["C"]
target_glyph.draw(target_svgpen_transformed)  
target_path = target_svgpen.getCommands()

# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
target_shapes = pydiffvg.from_svg_path(target_path)
# print(target_path)
target_path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
target_shape_groups = [target_path_group]
target_scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, target_shapes, target_shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *target_scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/optimize_vf/target.png', gamma=2.2)
target = img.clone()

# print("display_image('results/optimize_vf/target.png')")

# Variable font

start_variable_font = TTFont("/content/diffvg/font_data/variable/OpenSans.ttf")
start_font = mutator.instantiateVariableFont(start_variable_font, {"wght": 300, "wdth": 100})

# transformation
start_units_per_em = start_font['head'].unitsPerEm
start_ttf2em = transform.Identity.scale(1/start_units_per_em, 1/start_units_per_em)
start_svg_per_em=100
start_xform = transform.Identity.translate(0, start_svg_per_em).scale(start_svg_per_em, -start_svg_per_em).transform(start_ttf2em)

# glyph to svg
start_glyph_set = start_font.getGlyphSet()
start_svgpen = SVGPathPen(start_glyph_set)
start_svgpen_transformed = TransformPen(start_svgpen, start_xform)
start_glyph = start_glyph_set["C"]
start_glyph.draw(start_svgpen_transformed)  
start_path = start_svgpen.getCommands()

# Move the path to produce initial guess
# normalize points for easier learning rate
# noise = torch.FloatTensor(shapes[0].points.shape).uniform_(0.0, 1.0)
start_shapes = pydiffvg.from_svg_path(start_path)
start_points_n = start_shapes[0].points.clone()
start_points_n.requires_grad = True
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
start_shapes[0].points = start_points_n
start_path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = color)
start_shape_groups = [start_path_group]
start_scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, start_shapes, start_shape_groups)
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *start_scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_path/init.png', gamma=2.2)

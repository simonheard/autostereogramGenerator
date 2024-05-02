import datetime
import logging
from PIL import Image
import io
from flask import send_file
import numpy as np
# import matplotlib.pyplot as plt
# import skimage, skimage.io


# plt.rcParams['figure.dpi'] = 150

def process_request(request):
    # List of allowed origins
    allowed_origins = ['https://hede.wang/']

    # Get the origin of the incoming request
    origin = request.headers.get('Origin')
    
    # Set the logging level to DEBUG
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f'Request: {request}')


    # Check if the origin is in the list of allowed origins
    if origin in allowed_origins:
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Credentials': 'true'
        }
    else:
        # Optionally handle the disallowed origin case
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Credentials': 'true'
        }

    # Handle pre-flight requests
    if request.method == 'OPTIONS':
        # Pre-flight request contains no data, but needs the headers
        return ('', 204, headers)
    
    if request.method == 'GET':
        shape = request.args.get('shape', '128,128')
        shape_list = shape.split(',')
        if len(shape_list) != 2:
            return 'Shape format incorrect.'
        try:
            depthmap_shape = [int(shape_list[0]), int(shape_list[1])]
        except ValueError:
            return 'Shape format incorrect.'
        try:
            radius = int(request.args.get('radius', 20))
        except ValueError:
            return 'Radius format incorrect.'
        pattern_shape = request.args.get('pattern_shape', '16,16')
        pattern_shape_list = pattern_shape.split(',')
        if len(pattern_shape_list) != 2:
            return 'Pattern shape format incorrect.'
        try:
            pattern_shape = [int(pattern_shape_list[0]), int(pattern_shape_list[1])]
        except ValueError:
            return 'Pattern shape format incorrect.'
        try:
            levels = int(request.args.get('levels', 5))
        except ValueError:
            return 'Levels format incorrect.'
        try:
            shift_amplitude = float(request.args.get('shift_amplitude', 0.5))
        except ValueError:
            return 'Shift amplitude format incorrect.'
        invert = request.args.get('invert', 'n')
        if invert.lower() not in ['y', 'n']:
            return 'Invert format incorrect.'
        invert = True if invert.lower() == 'y' else False
        
        depthmap = create_circular_depthmap(shape=(depthmap_shape[0], depthmap_shape[1]), radius=radius)
        pattern = make_pattern(shape=(pattern_shape[0], pattern_shape[1]), levels=levels)
        autostereogram = make_autostereogram(depthmap, pattern, shift_amplitude=shift_amplitude, invert=invert)
        
        buf = io.BytesIO()
        img = Image.fromarray((autostereogram * 255).astype(np.uint8))
        img.save(buf, format='JPEG')
        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='autostereogram.jpeg'
            ), 200, headers
        
    elif request.method == 'POST':
        shape = request.args.get('shape', '128,128')
        shape_list = shape.split(',')
        if len(shape_list) != 2:
            return 'Shape format incorrect.'
        try:
            depthmap_shape = [int(shape_list[0]), int(shape_list[1])]
        except ValueError:
            return 'Shape format incorrect.'
        pattern_shape = request.args.get('pattern_shape', '16,16')
        pattern_shape_list = pattern_shape.split(',')
        if len(pattern_shape_list) != 2:
            return 'Pattern shape format incorrect.'
        try:
            pattern_shape = [int(pattern_shape_list[0]), int(pattern_shape_list[1])]
        except ValueError:
            return 'Pattern shape format incorrect.'
        try:
            levels = int(request.args.get('levels', 5))
        except ValueError:
            return 'Levels format incorrect.'
        try:
            shift_amplitude = float(request.args.get('shift_amplitude', 0.5))
        except ValueError:
            return 'Shift amplitude format incorrect.'
        invert = request.args.get('invert', 'n')
        if invert.lower() not in ['y', 'n']:
            return 'Invert format incorrect.'
        invert = True if invert.lower() == 'y' else False
        picture_shape = request.args.get('picture_shape', 'circle')
        if picture_shape not in ['circle', 'rectangle', 'triangle']:
            return 'Picture shape format incorrect.'
        if picture_shape == 'circle':
            try:
                radius = int(request.form.get('radius', 20))
            except ValueError:
                return 'Radius format incorrect.'
            depthmap = create_circular_depthmap(shape=(depthmap_shape[0], depthmap_shape[1]), radius=radius)
        elif picture_shape == 'rectangle':
            try:
                width = int(request.form.get('width', 20))
            except ValueError:
                return 'Width format incorrect.'
            try:
                height = int(request.form.get('height', 20))
            except ValueError:
                return 'Height format incorrect.'
            depthmap = create_rectangular_depthmap(shape=(depthmap_shape[0], depthmap_shape[1]), width=width, height=height)
        elif picture_shape == 'triangle':
            try:
                base = int(request.form.get('width', 20))
            except ValueError:
                return 'Base format incorrect.'
            try:
                height = int(request.form.get('height', 20))
            except ValueError:
                return 'Height format incorrect.'
            depthmap = create_triangular_depthmap(shape=(depthmap_shape[0], depthmap_shape[1]), base=base, height=height)
            
        pattern = make_pattern(shape=(pattern_shape[0], pattern_shape[1]), levels=levels)
        autostereogram = make_autostereogram(depthmap, pattern, shift_amplitude=shift_amplitude, invert=invert)
        
        buf = io.BytesIO()
        img = Image.fromarray((autostereogram * 255).astype(np.uint8))
        img.save(buf, format='JPEG')
        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='autostereogram.jpeg'
            ), 200, headers
        
    else:
        return 'Unsupported method', 405

def make_pattern(shape=(16, 16), levels=64):
    "Creates a pattern from gray values."
    pattern = np.random.randint(0, levels - 1, shape) / levels
    return pattern


def create_circular_depthmap(shape=(600, 800), center=None, radius=100, gradient=False):
    "Creates a circular depthmap, centered on the image."
    depthmap = np.zeros(shape, dtype=float)
    r = np.arange(depthmap.shape[0])
    c = np.arange(depthmap.shape[1])
    R, C = np.meshgrid(r, c, indexing='ij')
    if center is None:
        center = np.array([r.max() / 2, c.max() / 2])
    d = np.sqrt((R - center[0])**2 + (C - center[1])**2)
    if gradient:
        depthmap += 1 - d / radius
        depthmap = np.clip(depthmap, 0, 1)
    else:
        depthmap += (d < radius)
    return depthmap

def create_rectangular_depthmap(shape=(600, 800), center=None, width=200, height=100):
    "Creates a rectangular depthmap, centered on the image."
    depthmap = np.zeros(shape, dtype=float)
    r = np.arange(depthmap.shape[0])
    c = np.arange(depthmap.shape[1])
    R, C = np.meshgrid(r, c, indexing='ij')
    if center is None:
        center = np.array([r.max() / 2, c.max() / 2])
    d_x = np.abs(R - center[0])
    d_y = np.abs(C - center[1])
    depthmap += (d_x < width/2) & (d_y < height/2)
    return depthmap

def create_triangular_depthmap(shape=(600, 800), center=None, base=200, height=100):
    "Creates a triangular depthmap, centered on the image."
    depthmap = np.zeros(shape, dtype=float)
    r = np.arange(depthmap.shape[0])
    c = np.arange(depthmap.shape[1])
    R, C = np.meshgrid(r, c, indexing='ij')
    if center is None:
        center = np.array([r.max() / 2, c.max() / 2])
    d_x = np.abs(R - center[0])
    d_y = np.abs(C - center[1])
    depthmap += (d_x < height/2) & (d_y < base/2) & (d_y < -base/height * d_x + base/2)
    return depthmap

def normalize(depthmap):
    "Normalizes values of depthmap to [0, 1] range."
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap


def make_autostereogram(depthmap, pattern, shift_amplitude=0.1, invert=False):
    "Creates an autostereogram from depthmap and pattern."
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1 - depthmap
    autostereogram = np.zeros_like(depthmap, dtype=pattern.dtype)
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram

# def save_image(img, title=None, colorbar=False):
#     "Save an image"
#     plt.figure(figsize=(10, 10))
#     if len(img.shape) == 2:
#         i = skimage.io.imshow(img, cmap='gray')
#     else:
#         i = skimage.io.imshow(img)
#     if colorbar:
#         plt.colorbar(i, shrink=0.5, label='depth')
#     if title:
#         plt.title(title)
#     plt.tight_layout()
#     datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
#     filename = f"image_{datetime_str}.png"
#     plt.savefig(filename)
#     print(f"Image saved as {filename}.")

# if __name__ == "__main__":

#     print("Welcome to the autostereogram generator!")

#     print("Step 1: Depth Map")
#     while True:
#         input_str = input("Please enter the shape (2 integers) for the depth map, separated by a comma:")
#         input_list = input_str.split(',')
#         if len(input_list) != 2:
#             print("Format incorrect. Please try again.")
#             continue
#         try:
#             depthmap_shape = [int(input_list[0]), int(input_list[1])]
#             break
#         except ValueError:
#             print("Format incorrect. Please try again.")
#             continue

#     while True:
#         input_str = input("Please enter the radius (1 integer) for the depth map:")
#         try:
#             radius = int(input_str)
#             break
#         except ValueError:
#             print("Format incorrect. Please try again.")
#             continue

#     print("Step 2: Pattern")
#     while True:
#         input_str = input("Please enter the shape (2 integers) for the pattern, separated by a comma:")
#         input_list = input_str.split(',')
#         if len(input_list) != 2:
#             print("Format incorrect. Please try again.")
#             continue
#         try:
#             pattern_shape = [int(input_list[0]), int(input_list[1])]
#             break
#         except ValueError:
#             print("Format incorrect. Please try again.")
#             continue

#     while True:
#         input_str = input("Please enter the number of levels (1 integer) for the pattern:")
#         try:
#             levels = int(input_str)
#             break
#         except ValueError:
#             print("Format incorrect. Please try again.")
#             continue

#     print("Step 3: Autostereogram")
#     while True:
#         input_str = input("Please enter the shift amplitude (1 float) for the pattern, separated by a comma:")
#         try:
#             shift_amplitude = float(input_str)
#             break
#         except ValueError:
#             print("Format incorrect. Please try again.")
#             continue
#     while True:
#         input_str = input("Please decide whether to invert the autostereogram (y/Y for yes, n/N for no):")
#         if len(input_str) != 1 or input_str not in ['y', 'Y', 'n', 'N']:
#             print("Format incorrect. Please try again.")
#             continue
#         invert = True if input_str.lower() == "y" else False
#         break

#     depthmap = create_circular_depthmap(shape=(depthmap_shape[0], depthmap_shape[1]), radius=radius)

#     pattern = make_pattern(shape=(pattern_shape[0], pattern_shape[1]), levels=levels)

#     autostereogram = make_autostereogram(depthmap, pattern, shift_amplitude=shift_amplitude, invert=invert)

#     title = f"Depth map: shape = {depthmap_shape[0]} x {depthmap_shape[1]} , radius = {radius} \n Pattern: shape = {pattern_shape[0]} x {pattern_shape[1]} , levels = {levels} \n Autostereogram: shift amplitude = {shift_amplitude}, invert = {invert}"
#     save_image(autostereogram, title=title)



# depthmap = create_rectangular_depthmap()
# plt.imshow(depthmap, cmap="gray")
# plt.axis("off")
# plt.show()
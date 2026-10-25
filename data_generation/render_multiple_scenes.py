# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
import numpy as np
from datetime import datetime as dt
from collections import Counter
from utils import get_3x4_RT_matrix_from_blender

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
	import bpy, bpy_extras
	from mathutils import Vector
except ImportError as e:
	INSIDE_BLENDER = False
if INSIDE_BLENDER:
	try:
		import utils
	except ImportError as e:
		print("\nERROR")
		print("Running render_images.py from Blender and cannot import utils.py.")
		print("You may need to add a .pth file to the site-packages of Blender's")
		print("bundled python with a command like this:\n")
		print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
		print("\nWhere $BLENDER is the directory where Blender is installed, and")
		print("$VERSION is your Blender version (such as 2.78).")
		sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--no_background', action='store_true')
parser.add_argument('--n_scenes', type=int, default=2)
parser.add_argument('--start_scene', type=int, default=0)
parser.add_argument('--output_dir', default='../output/debug_57obj')
parser.add_argument('--scenes_dir', default='blend_scenes/')
parser.add_argument('--n_az', type=int, default=2)
parser.add_argument('--multi_dist', action='store_true')  # use 3 distances
parser.add_argument('--properties_json', default='data/properties.json',
					help="JSON file defining objects, materials, sizes, and colors. " +
						 "The \"colors\" field maps from CLEVR color names to RGB values; " +
						 "The \"sizes\" field maps from CLEVR size names to scalars used to " +
						 "rescale object models; the \"materials\" and \"shapes\" fields map " +
						 "from CLEVR material and shape names to .blend files in the " +
						 "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
					help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
					help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
					help="Optional path to a JSON file mapping shape names to a list of " +
						 "allowed color names for that shape. This allows rendering images " +
						 "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
					help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
					help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
					help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
					help="Along all cardinal directions (left, right, front, back), all " +
						 "objects will be at least this distance apart. This makes resolving " +
						 "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
					help="All objects will have at least this many visible pixels in the " +
						 "final rendered images; this ensures that no objects are fully " +
						 "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
					help="The number of times to try placing an object before giving up and " +
						 "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
					help="The index at which to start for numbering rendered images. Setting " +
						 "this to non-zero values allows you to distribute rendering across " +
						 "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
					help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
					help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--output_image_dir', default='../output/images/',
					help="The directory where output images will be stored. It will be " +
						 "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
					help="The directory where output JSON scene structures will be stored. " +
						 "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
					help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
					help="The directory where blender scene files will be stored, if the " +
						 "user requested that these files be saved using the " +
						 "--save_blendfiles flag; in this case it will be created if it does " +
						 "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
					help="Setting --save_blendfiles 1 will cause the blender scene file for " +
						 "each generated image to be stored in the directory specified by " +
						 "the --output_blend_dir flag. These files are not saved by default " +
						 "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
					help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
					default="Creative Commons Attribution (CC-BY 4.0)",
					help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
					help="String to store in the \"date\" field of the generated JSON file; " +
						 "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
					help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
						 "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
						 "to work.")
parser.add_argument('--width', default=160, type=int,
					help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=120, type=int,
					help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
					help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
					help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
					help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
					help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
					help="The number of samples to use when rendering. Larger values will " +
						 "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
					help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
					help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
					help="The tile size to use for rendering. This should not affect the " +
						 "quality of the rendered image but may affect the speed; CPU-based " +
						 "rendering may achieve better performance using smaller tile sizes " +
						 "while larger tile sizes may be optimal for GPU-based rendering.")


def cart2sph(x, y, z):
	hxy = np.hypot(x, y)
	r = np.hypot(hxy, z)
	el = np.arctan2(z, hxy)
	az = np.arctan2(y, x)
	return az, el, r


def sph2cart(az, el, r):
	rcos_theta = r * np.cos(el)
	x = rcos_theta * np.cos(az)
	y = rcos_theta * np.sin(az)
	z = r * np.sin(el)
	return x, y, z


def main(args):
	idx_img = args.start_scene * args.n_az
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	print('idx_img:{}'.format(idx_img))
	for i in range(args.start_scene, args.start_scene+args.n_scenes):
		scene_name = '{:03d}'.format(i)
		print('scene_name:{}'.format(scene_name))

		# Load the main blendfile
		bpy.ops.wm.open_mainfile(filepath=os.path.join(args.scenes_dir, scene_name + '.blend'))
		fp = bpy.data.objects['Ground'].active_material.node_tree.nodes['Image Texture.004'].image.filepath
		fp = fp[5:]
		bpy.data.objects['Ground'].active_material.node_tree.nodes['Image Texture.004'].image.filepath = fp
		print('bg texture path:')
		print(bpy.data.objects['Ground'].active_material.node_tree.nodes['Image Texture.004'].image.filepath)
		print('loaded main blendfile')
		# Load materials
		utils.load_materials(args.material_dir)
		if args.no_background:
			print('do not use background plane.')
			bpy.data.objects['Ground'].hide_render = True
			bpy.data.objects['Lamp_Back'].location[1] = 5.
			bpy.data.objects['Lamp_Back'].location[2] = 3.

		# Set render arguments so we can get pixel coordinates later.
		# We use functionality specific to the CYCLES renderer so BLENDER_RENDER
		# cannot be used.
		render_args = bpy.context.scene.render
		render_args.engine = "CYCLES"
		render_args.resolution_x = args.width
		render_args.resolution_y = args.height
		if args.width == 256:
			render_args.pixel_aspect_x = 320 / 240.
		render_args.resolution_percentage = 100
		render_args.tile_x = args.render_tile_size
		render_args.tile_y = args.render_tile_size
		if args.use_gpu == 1:
			# Blender changed the API for enabling CUDA at some point
			if bpy.app.version < (2, 78, 0):
				bpy.context.user_preferences.system.compute_device_type = 'CUDA'
				bpy.context.user_preferences.system.compute_device = 'CUDA_0'
			else:
				cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
				cycles_prefs.compute_device_type = 'CUDA'

		# Some CYCLES-specific stuff
		bpy.data.worlds['World'].cycles.sample_as_light = True
		bpy.context.scene.cycles.blur_glossy = 2.0
		bpy.context.scene.cycles.samples = args.render_num_samples
		bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
		bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
		bpy.context.scene.camera.data.clip_start = 6
		if args.use_gpu == 1:
			bpy.context.scene.cycles.device = 'GPU'

		x, y, z = bpy.data.objects['Camera'].location
		az, el, r = cart2sph(x, y, z)

		delta_az_list = [x / args.n_az * np.pi * 2 for x in range(args.n_az)]
		az_range = np.pi / args.n_az
		# az_range = 0
		delta_dist_list = [1.2, 1, 0.8] if args.multi_dist else [1.1, ]
		print('in main loop')
		for i_az, delta_az in enumerate(delta_az_list):
			az_ = az + delta_az + (random.random() - 0.5) * az_range
			el_ = el + np.pi / 15
			for i_dist, delta_dist in enumerate(delta_dist_list):
				r_ = r * delta_dist
				x_, y_, z_ = sph2cart(az_, el_, r_)
				bpy.data.objects['Camera'].location[0] = x_
				bpy.data.objects['Camera'].location[1] = y_
				bpy.data.objects['Camera'].location[2] = z_
				if not args.multi_dist:
					output_image = '{}/{:05d}_sc{:04d}_az{:02d}.png'.format(args.output_dir, idx_img, i, i_az)
				else:
					output_image = '{}/{:05d}_sc{:04d}_az{:02d}_dist{:01d}.png'.format(args.output_dir, idx_img, i,
																								i_az, i_dist)
				render_args.filepath = output_image

				print('img path: {}'.format(render_args.filepath))
				print('x_, y_, z_: {}, {}, {}'.format(x_, y_, z_))
				print('camera location: {}'.format(bpy.data.objects['Camera'].location))

				while True:
					try:
						bpy.ops.render.render(write_still=True)
						break
					except Exception as e:
						print(e)
				RT = get_3x4_RT_matrix_from_blender(bpy.data.objects['Camera'])
				RT = np.array(RT)
				world2cam = np.eye(4)
				world2cam[:3, :] = RT
				cam2world = np.linalg.inv(world2cam)
				if not args.multi_dist:
					np.savetxt(
						'{}/{:05d}_sc{:04d}_az{:02d}_RT.txt'.format(args.output_dir, idx_img, i, i_az), cam2world,
						fmt='%.5f')
				else:
					np.savetxt(
						'{}/{:05d}_sc{:04d}_az{:02d}_dist{:01d}_RT.txt'.format(args.output_dir, idx_img, i,
																						i_az, i_dist), cam2world,
						fmt='%.5f')
				cam_pos = cam2world[:3, 3]
				print('camera matrix pos: {}'.format(cam_pos))

				cam_matrix = np.array(bpy.data.objects['Camera'].matrix_world)
				np.savetxt(
					'{}/{:05d}_sc{:04d}_az{:02d}_pixel_nerf_matrix.txt'.format(args.output_dir, idx_img, i, i_az), cam_matrix, fmt='%.5f')

				az_ori = az_ + math.pi  # since cam faces world origin, its orientation azimuth differs by pi
				az_rot_mat_2d = np.array([[math.cos(az_ori), - math.sin(az_ori)],
										  [math.sin(az_ori), math.cos(az_ori)]])
				az_rot_mat = np.eye(3)
				az_rot_mat[:2, :2] = az_rot_mat_2d

				np.savetxt(
					'{}/{:05d}_sc{:04d}_az{:02d}_azi_rot.txt'.format(args.output_dir, idx_img, i, i_az), az_rot_mat, fmt='%.5f')

				sys.stdout.flush()
				idx_img += 1


if __name__ == '__main__':
	if INSIDE_BLENDER:
		# Run normally
		argv = utils.extract_args()
		args = parser.parse_args(argv)
		main(args)
	elif '--help' in sys.argv or '-h' in sys.argv:
		parser.print_help()
	else:
		print('This script is intended to be called from blender like this:')
		print()
		print('blender --background --python render_images.py -- [args]')
		print()
		print('You can also run as a standalone python script to view all')
		print('arguments like this:')
		print()
		print('python render_images.py --help')

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os
import bpy, bpy_extras
from mathutils import Matrix, Vector
import math

"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
	"""
	Pull out command-line arguments after "--". Blender ignores command-line flags
	after --, so this lets us forward command line arguments from the blender
	invocation to our own script.
	"""
	if input_argv is None:
		input_argv = sys.argv
	output_argv = []
	if '--' in input_argv:
		idx = input_argv.index('--')
		output_argv = input_argv[(idx + 1):]
	return output_argv


def parse_args(parser, argv=None):
	return parser.parse_args(extract_args(argv))


def delete_object(obj):
	""" Delete a specified blender object """
	for o in bpy.data.objects:
		o.select = False
	obj.select = True
	bpy.ops.object.delete()


def get_camera_coords(cam, pos):
	"""
	For a specified point, get both the 3D coordinates and 2D pixel-space
	coordinates of the point from the perspective of the camera.

	Inputs:
	- cam: Camera object
	- pos: Vector giving 3D world-space position

	Returns a tuple of:
	- (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
	  in the range [-1, 1]
	"""
	scene = bpy.context.scene
	x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
	scale = scene.render.resolution_percentage / 100.0
	w = int(scale * scene.render.resolution_x)
	h = int(scale * scene.render.resolution_y)
	px = int(round(x * w))
	py = int(round(h - y * h))
	return (px, py, z)


def set_layer(obj, layer_idx):
	""" Move an object to a particular layer """
	# Set the target layer to True first because an object must always be on
	# at least one layer.
	obj.layers[layer_idx] = True
	for i in range(len(obj.layers)):
		obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, horizon_scale, vertical_scale, loc, theta=0, z=None):
	"""
	Load an object from a file. We assume that in the directory object_dir, there
	is a file named "$name.blend" which contains a single object named "$name"
	that has unit size and is centered at the origin.

	- scale: scalar giving the size that the object should be in the scene
	- loc: tuple (x, y) giving the coordinates on the ground plane where the
	  object should be placed.
	"""
	# First figure out how many of this object are already in the scene so we can
	# give the new object a unique name
	count = 0
	for obj in bpy.data.objects:
		if obj.name.startswith(name):
			count += 1

	filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
	bpy.ops.wm.append(filename=filename)

	# Give it a new name to avoid conflicts
	new_name = '%s_%d' % (name, count)
	bpy.data.objects[name].name = new_name

	# Set the new object as active, then rotate, scale, and translate it
	x, y = loc
	z = horizon_scale if z is None else z
	bpy.context.scene.objects.active = bpy.data.objects[new_name]
	bpy.context.object.rotation_euler[2] = theta
	bpy.ops.transform.resize(value=(horizon_scale, horizon_scale, vertical_scale))
	bpy.ops.transform.translate(value=(x, y, z))

def add_object_obj(filename, obj_name, horizon_scale, vertical_scale, loc, theta=0, z=None):
	"""
	Load an object from an OBJ file. We assume "$filename" contains a single object named "$obj_name"
	that has unit size and is centered at the origin.

	- scale: scalar giving the size that the object should be in the scene
	- loc: tuple (x, y) giving the coordinates on the ground plane where the
	  object should be placed.
	"""
	# First figure out how many of this object are already in the scene so we can
	# give the new object a unique name
	count = 0
	for obj in bpy.data.objects:
		if obj.name.startswith(obj_name):
			count += 1

	bpy.ops.import_scene.obj(filepath=filename)

	# print('bpy.context.selected_objects:{}'.format(bpy.context.selected_objects))
	obj_name_loaded = bpy.context.selected_objects[0].name

	# print('obj_name_loaded: {}'.format(obj_name_loaded))
	# print('bpy.data.objects names:')
	# print(list(bpy.data.objects))

	# Give it a new name to avoid conflicts
	new_name = '%s_%d' % (obj_name, count)
	bpy.data.objects[obj_name_loaded].name = new_name

	# Set the new object as active, then rotate, scale, and translate it
	x, y = loc
	z = horizon_scale if z is None else z
	bpy.context.scene.objects.active = bpy.data.objects[new_name]
	bpy.context.object.rotation_euler[2] = theta
	bpy.ops.transform.resize(value=(horizon_scale, horizon_scale, vertical_scale))
	bpy.ops.transform.translate(value=(x, y, z))

def load_materials(material_dir):
	"""
	Load materials from a directory. We assume that the directory contains .blend
	files with one material each. The file X.blend has a single NodeTree item named
	X; this NodeTree item must have a "Color" input that accepts an RGBA value.
	"""
	for fn in os.listdir(material_dir):
		if not fn.endswith('.blend'): continue
		name = os.path.splitext(fn)[0]
		filepath = os.path.join(material_dir, fn, 'NodeTree', name)
		bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
	"""
	Create a new material and assign it to the active object. "name" should be the
	name of a material that has been previously loaded using load_materials.
	"""
	# Figure out how many materials are already in the scene
	mat_count = len(bpy.data.materials)

	# Create a new material; it is not attached to anything and
	# it will be called "Material"
	bpy.ops.material.new()

	# Get a reference to the material we just created and rename it;
	# then the next time we make a new material it will still be called
	# "Material" and we will still be able to look it up by name
	mat = bpy.data.materials['Material']
	mat.name = 'Material_%d' % mat_count

	# Attach the new material to the active object
	# Make sure it doesn't already have materials
	obj = bpy.context.active_object
	assert len(obj.data.materials) == 0
	obj.data.materials.append(mat)

	# Find the output node of the new material
	output_node = None
	for n in mat.node_tree.nodes:
		if n.name == 'Material Output':
			output_node = n
			break

	# Add a new GroupNode to the node tree of the active material,
	# and copy the node tree from the preloaded node group to the
	# new group node. This copying seems to happen by-value, so
	# we can create multiple materials of the same type without them
	# clobbering each other
	group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
	group_node.node_tree = bpy.data.node_groups[name]

	# Find and set the "Color" input of the new group node
	for inp in group_node.inputs:
		if inp.name in properties:
			inp.default_value = properties[inp.name]

	# Wire the output of the new group node to the input of
	# the MaterialOutput node
	mat.node_tree.links.new(
		group_node.outputs['Shader'],
		output_node.inputs['Surface'],
	)


def add_ground(name):
	"""
	Add ground material to the base
	"""

	# Create a new material; it is not attached to anything and
	# it will be called "Material"
	bpy.ops.material.new()

	# Get a reference to the material we just created and rename it;
	# then the next time we make a new material it will still be called
	# "Material" and we will still be able to look it up by name
	mat = bpy.data.materials['Material']
	mat.name = 'Material_ground'

	obj = bpy.data.objects['Ground']
	# Attach the new material to the active object
	# Make sure it doesn't already have materials
	assert len(obj.data.materials) == 0
	obj.data.materials.append(mat)

	# Find the output node of the new material
	output_node = None
	for n in mat.node_tree.nodes:
		if n.name == 'Material Output':
			output_node = n
			break

	# Add a new GroupNode to the node tree of the active material,
	# and copy the node tree from the preloaded node group to the
	# new group node. This copying seems to happen by-value, so
	# we can create multiple materials of the same type without them
	# clobbering each other
	group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
	group_node.node_tree = bpy.data.node_groups[name]

	# Wire the output of the new group node to the input of
	# the MaterialOutput node
	mat.node_tree.links.new(
		group_node.outputs['Shader'],
		output_node.inputs['Surface'],
	)

	disp = False
	for o in group_node.outputs:
		if o.name == "Displacement":
			disp = True

	if disp:
		mat.node_tree.links.new(
			group_node.outputs['Displacement'],
			output_node.inputs['Displacement'],
		)


# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
	if sensor_fit == 'VERTICAL':
		return sensor_y
	return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
	if sensor_fit == 'AUTO':
		if size_x >= size_y:
			return 'HORIZONTAL'
		else:
			return 'VERTICAL'
	return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
	if camd.type != 'PERSP':
		raise ValueError('Non-perspective cameras not supported')
	scene = bpy.context.scene
	f_in_mm = camd.lens
	scale = scene.render.resolution_percentage / 100
	resolution_x_in_px = scale * scene.render.resolution_x
	resolution_y_in_px = scale * scene.render.resolution_y
	sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
	sensor_fit = get_sensor_fit(
		camd.sensor_fit,
		scene.render.pixel_aspect_x * resolution_x_in_px,
		scene.render.pixel_aspect_y * resolution_y_in_px
	)
	pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
	if sensor_fit == 'HORIZONTAL':
		view_fac_in_px = resolution_x_in_px
	else:
		view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
	pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
	s_u = 1 / pixel_size_mm_per_px
	s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

	# Parameters of intrinsic calibration matrix K
	u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
	v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
	skew = 0  # only use rectangular pixels

	K = Matrix(
		((s_u, skew, u_0),
		 (0, s_v, v_0),
		 (0, 0, 1)))
	return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
	# bcam stands for blender camera
	R_bcam2cv = Matrix(
		((1, 0, 0),
		 (0, -1, 0),
		 (0, 0, -1)))

	# Transpose since the rotation is object rotation,
	# and we want coordinate rotation
	# R_world2bcam = cam.rotation_euler.to_matrix().transposed()
	# T_world2bcam = -1*R_world2bcam * location
	#
	# Use matrix_world instead to account for all constraints
	location, rotation = cam.matrix_world.decompose()[0:2]
	R_world2bcam = rotation.to_matrix().transposed()

	# Convert camera location to translation vector used in coordinate changes
	# T_world2bcam = -1*R_world2bcam * cam.location
	# Use location from matrix_world to account for constraints:
	T_world2bcam = -1 * R_world2bcam * location

	# Build the coordinate transform matrix from world to computer vision camera
	R_world2cv = R_bcam2cv * R_world2bcam
	T_world2cv = R_bcam2cv * T_world2bcam

	# put into 3x4 matrix
	RT = Matrix((
		R_world2cv[0][:] + (T_world2cv[0],),
		R_world2cv[1][:] + (T_world2cv[1],),
		R_world2cv[2][:] + (T_world2cv[2],)
	))
	return RT


def get_3x4_P_matrix_from_blender(cam):
	K = get_calibration_matrix_K_from_blender(cam.data)
	RT = get_3x4_RT_matrix_from_blender(cam)
	return K * RT, K, RT


# ----------------------------------------------------------
if __name__ == "__main__":
	# Insert your camera name here
	cam = bpy.data.objects['Camera']
	P, K, RT = get_3x4_P_matrix_from_blender(cam)
	print("K")
	print(K)
	print("RT")
	print(RT)
	print("P")
	print(P)

	print("==== 3D Cursor projection ====")
	pc = P * bpy.context.scene.cursor.location
	pc /= pc[2]
	print("Projected cursor location")
	print(pc)

	# Bonus code: save the 3x4 P matrix into a plain text file
	# Don't forget to import numpy for this
	# nP = numpy.matrix(P)
	# numpy.savetxt("/tmp/P3x4.txt", nP)  # to select precision, use e.g. fmt='%.2f'

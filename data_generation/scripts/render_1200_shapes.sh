#!/bin/bash
blender-2.78c/blender --background --python render_multiple_scenes.py -- \
--use_gpu 1 --n_scenes 1000 --start_scene 0 --scenes_dir 'blend_scenes/1200shape_50bg' \
--output_dir 'datasets/1200shape_50bg' --n_az 4 --width 256 --height 256
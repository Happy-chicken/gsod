#!/bin/bash
blender-2.78c/blender --background --python generate_blend_files.py -- \
--n_scenes 5000 --output_dir 'blend_scenes/1200shape_50bg' --min_objects 4 --max_objects 4 \
--diffuse_only --use_gpu 1 --use_background --shape_dir 'data/example_shapes' \
--min_dist 0.5 --margin 0 --height 240 --width 320 --properties_json 'data/properties_1000chair.json' --chair_only
#!/bin/bash
blender-2.78c/blender --background --python generate_blend_files.py -- \
--n_scenes 500 --output_dir 'blend_scenes/1200shape_50bg_test' --min_objects 4 --max_objects 4 \
--diffuse_only --use_gpu 0 --use_background --shape_dir 'data/example_shapes' \
--min_dist 0.5 --margin 0 --height 240 --width 320 --properties_json 'data/properties_1000chair.json' --chair_only
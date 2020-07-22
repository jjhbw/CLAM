import os
import time
import argparse
from wsi_core.WholeSlideImage import WholeSlideImage, StitchPatches

parser = argparse.ArgumentParser(description='Segmentation and patching')
parser.add_argument('--input_slide', type = str, help='Path to input WSI file')
parser.add_argument('--output_dir', type = str, help='Directory to save output data')
args = parser.parse_args()

# TODO: this is a mess. Re-evaluate which parameters are even required.
seg_params = {'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False}
filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
vis_params = {'line_thickness': 250}
patch_params = {
    'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt',
    'patch_level': 0, 'patch_size': 256, 'step_size': 256, 'custom_downsample': 1    
}

# Derive the slide ID from its name
slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))

# Make the output directory if it doesnt exist yet
os.makedirs(args.output_dir, exist_ok=True)

# Read the slide
WSI_object = WholeSlideImage(args.input_slide, hdf5_file=None)

# Determine the best level to determine the segmentation on
seg_level = WSI_object.getOpenSlide().get_best_level_for_downsample(64)

w, h = WSI_object.level_dim[seg_level] 
if w * h > 1e8:
    raise Exception('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))

# Segment tissue (this method modifies in-place...)
print('Segmenting tissue...')
start_time = time.time()
WSI_object.segmentTissue(**seg_params, seg_level=seg_level, filter_params=filter_params)
print(f"Segmentation finished in {time.time() - start_time}")

# Save the resulting mask
mask = WSI_object.visWSI(**vis_params, vis_level=seg_level)
mask.save(os.path.join(args.output_dir, slide_id + '_tissue_mask.png'))

# Patching
## path needs to be a directory... poor API design
patch_save_dir = args.output_dir
print('Creating bag of patches...')
start_time = time.time()
WSI_object.createPatches_bag_hdf5(**patch_params, save_path=patch_save_dir, save_coord=True)
print(f"Patch bag creation finished in {time.time() - start_time}")

# Stitching
## This path needs to reference the exact file... Note that the actual filename is decided in createPatches_bag_hdf5 (IMO poor design)
patch_file_path = os.path.join(patch_save_dir, slide_id + '.h5')
start_time = time.time()
heatmap = StitchPatches(patch_file_path, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
heatmap.save(os.path.join(args.output_dir, slide_id + '_stitched.png'))
print(f"Stitching finished in {time.time() - start_time}")

# Rename the file containing the patches to ensure we can easily
# distinguish incomplete bags of patches (.h5 files) from complete ones in case a job fails.
os.rename(patch_file_path, os.path.join(patch_save_dir, slide_id + '_patches.h5'))
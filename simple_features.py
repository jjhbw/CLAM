
import argparse
import os
import torch
import time
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
from datasets.dataset_h5 import Whole_Slide_Bag
from utils.utils import collate_features
import h5py

def save_hdf5(output_dir, asset_dict, mode='a'):
	file = h5py.File(output_dir, mode)

	for key, val in asset_dict.items():
		data_shape = val.shape
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
			dset[:] = val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis=0)
			dset[-data_shape[0]:] = val  

	file.close()
	return output_dir

parser = argparse.ArgumentParser(description='Feature extraction from patches')
parser.add_argument('--input_patches', type = str, help='Path to input patches (in .h5 format)')
parser.add_argument('--output_dir', type = str, help='Where to save the resulting features file')
args = parser.parse_args()

input_file = args.input_patches
input_file_base, _ = os.path.splitext(os.path.basename(args.input_patches))
output_file = os.path.join(args.output_dir, input_file_base + '_features.h5')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the model
model = resnet50_baseline(pretrained=True)
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.eval()

# TODO: make n workers configurable?
kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
loader = DataLoader(
    dataset=Whole_Slide_Bag(file_path=input_file, pretrained=True),
    batch_size=512,
    collate_fn=collate_features,
    **kwargs,
)

print('Processing {}: total of {} batches'.format(input_file, len(loader)))
start_time = time.time()

with torch.no_grad():
	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		print('Processing batch {}/{}...'.format(count, len(loader)))
		batch = batch.to(device, non_blocking=True)
		features = model(batch).cpu().numpy()
		save_hdf5(output_file, {'features': features, 'coords': coords}, mode=mode)
		mode = 'a'
print(f"Finished in {time.time() - start_time}")

# TODO: save as .pt instead of h5. Do we care?
# output_path_pt = os.path.join(args.output_dir, input_file_base+'.pt')
# file = h5py.File(output_file_path, "r")
# features = file['features'][:]
# features = torch.from_numpy(features)
# torch.save(features, output_path_pt)

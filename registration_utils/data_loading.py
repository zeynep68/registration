import h5py


def load_images(data, section_id, bf_pyramid_lvl, mod_pyramid_lvl):
	bf_mask_path = data['BF-Mask'][section_id][0]
	bf_mask_path = bf_mask_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	mod_mask_path = data['MOD-Mask'][section_id][0]
	mod_mask_path = mod_mask_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	bf_path = data['Blockface'][section_id][0]
	bf_path = bf_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	trans_path = data['Transmittance'][section_id][0]
	trans_path = trans_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	dir_path = data['Direction'][section_id][0]
	dir_path = dir_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	ret_path = data['Retardation'][section_id][0]
	ret_path = ret_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	bf_mask = h5py.File(bf_mask_path, 'r')['pyramid'][bf_pyramid_lvl]
	mod_mask = h5py.File(mod_mask_path, 'r')['pyramid'][mod_pyramid_lvl]
	bf = h5py.File(bf_path, 'r')['pyramid'][bf_pyramid_lvl]

	trans = h5py.File(trans_path, 'r')['pyramid'][mod_pyramid_lvl]
	direction = h5py.File(dir_path, 'r')['pyramid'][mod_pyramid_lvl]
	ret = h5py.File(ret_path, 'r')['pyramid'][mod_pyramid_lvl]

	return {'Transmittance': trans, 'Direction': direction, 'Retardation': ret, 'BF-Mask': bf_mask, 'Blockface': bf, 'MOD-Mask': mod_mask}

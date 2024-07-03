import json
import pysolr


def get_blockface_mask_query(brain_id):
	return f"brain_id: {brain_id} AND image_modality:Mask AND data_source:BF"


def get_blockface_query(brain_id):
	return f"brain_id: {brain_id} AND image_modality:Raw AND data_source:BF AND registered:1"


def get_pli_query(brain_id, mode):
	return f"brain_id: {brain_id} AND image_modality:{mode} AND stitched:1 AND focus_level:0 AND tilt_amplitude:0 AND section_roi:Complete AND data_source:LMP1"


def get_pli_mask_query(brain_id):
	return f"brain_id: {brain_id} AND image_modality:Mask AND section_roi:Complete AND stitched:1 AND focus_level:0 AND tilt_amplitude:0 AND data_source:LMP1"


def split_by_section_id(results, measurement_time=True):
	res = {}
	
	for i in results:
		path = i['path']
		timestamp = i['measurement_time'] if measurement_time else None

		section_id = i['section_id']


		if section_id in res.keys():
			res[section_id].append((path, timestamp))
		else:
			res[section_id] = [(path, timestamp)]
			
	return res


def get_first_timestamp(results):
	""" result (dict): For each section_id [key] -> tuple of (path, measurement_time) [value]
	"""
	for s in results.keys():
		results[s] = sorted(results[s], key=lambda x: x[1])[0]

	return results


def assert_each_section_available_in_each_modality(results):
	# assume 3 modalities (transmittance, retardation, direction)

	k1 = set(results["transmittance"].keys())
	k2 = set(results["retardation"].keys())
	k3 = set(results["direction"].keys())

	assert k1 == k2, 'Number of sections not equal!'
	assert k2 == k3, 'Number of sections not equal!'


def get_first_version(results):
	for s in results:
		results[s].sort()
		results[s] = results[s][0]

	return results


def filter_sections(blockface_masks, blockfaces, pli_masks, transmittance):
	""" Keep only those sections available in each modality 
	"""

	blockface_masks = {k: v for k, v in blockface_masks.items() if k in transmittance}		
	blockfaces = {k: v for k, v in blockfaces.items() if k in transmittance}		
	pli_masks = {k: v for k, v in pli_masks.items() if k in transmittance}		

	return blockface_masks, blockfaces, pli_masks


def main(brain_id, output_path, 
		 pli_modes=["transmittance", "retardation", "direction"]):
	solr = pysolr.Solr("http://ime262.ime.kfa-juelich.de:8984/solr/inm_metadata/", 
					   auth=("plibda_user", "plibda123"), timeout=4.2,)
	pli = {}

	blockface_masks = solr.search(get_blockface_mask_query(brain_id=brain_id))
	print('1:',len(blockface_masks))
	blockface_masks = split_by_section_id(blockface_masks, measurement_time=False)
	pli['blockface_masks'] = get_first_version(blockface_masks)

	blockfaces = solr.search(get_blockface_query(brain_id=brain_id))	
	print('2:',len(blockface))
	blockfaces = split_by_section_id(blockfaces, measurement_time=False)
	pli['blockfaces'] = get_first_version(blockfaces)

	pli_masks = solr.search(get_pli_mask_query(brain_id=brain_id))
	print('3:',len(pli_masks))
	pli_masks = split_by_section_id(pli_masks, measurement_time=False)
	pli['pli_masks'] = get_first_version(pli_masks)

	for mode in pli_modes:
		results = solr.search(get_pli_query(brain_id=brain_id, mode=mode))

		pli[mode] = get_first_timestamp(split_by_section_id(results))

	blockface_masks, blockfaces, pli_masks  = filter_sections(blockface_masks, blockfaces, pli_masks, pli["transmittance"])
	
	assert_each_section_available_in_each_modality(pli)

	for mode in pli:
		print(mode, len(pli[mode]))

	with open(output_path, 'w') as file:
		json.dump(pli, file, indent=4)


if __name__ == "__main__":
	main(brain_id="PE-2021-00981-H", output_path="pli_big_brain_paths.json")

import json
import pysolr


def get_bf_mask_query(brain_id):
	return f"brain_id: {brain_id} AND image_modality:Mask AND data_source:BF"


def get_blockface_query(brain_id):
	return f"brain_id: {brain_id} AND image_modality:Raw AND data_source:BF AND registered:1"


def get_modality_query(brain_id, modality):
	return f"brain_id: {brain_id} AND image_modality:{modality} AND stitched:1 AND focus_level:0 AND tilt_amplitude:0 AND section_roi:Complete AND data_source:LMP1"


def get_modality_mask_query(brain_id):
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

	k1 = set(results["Transmittance"].keys())
	k2 = set(results["Retardation"].keys())
	k3 = set(results["Direction"].keys())

	assert k1 == k2, 'Number of sections not equal!'
	assert k2 == k3, 'Number of sections not equal!'


def get_first_version(results):
	for s in results:
		results[s].sort()
		results[s] = results[s][0]

	return results


def filter_sections(bf_masks, blockfaces, mod_masks, modalities):
	""" Keep only those sections available in each modality 
	"""

	bf_masks = {key: value for key, value in bf_masks.items() if key in modalities["Transmittance"]}		
	blockfaces = {key: value for key, value in blockfaces.items() if key in modalities["Transmittance"]}		
	mod_masks = {key: value for key, value in mod_masks.items() if key in modalities["Transmittance"]}		

	return bf_masks, blockfaces, mod_masks


def main(brain_id="PE-2021-00981-H", modes=["Transmittance", "Retardation", "Direction"]):
	solr = pysolr.Solr("http://ime262.ime.kfa-juelich.de:8984/solr/inm_metadata/", auth=("plibda_user", "plibda123"), timeout=4.2,)

	bf_masks = solr.search(get_bf_mask_query(brain_id=brain_id))
	bf_masks = split_by_section_id(bf_masks, measurement_time=False)
	bf_masks = get_first_version(bf_masks)

	blockfaces = solr.search(get_blockface_query(brain_id=brain_id))	
	blockfaces = split_by_section_id(blockfaces, measurement_time=False)
	blockfaces = get_first_version(blockfaces)

	mod_masks = solr.search(get_modality_mask_query(brain_id=brain_id))
	mod_masks = split_by_section_id(mod_masks, measurement_time=False)
	mod_masks = get_first_version(mod_masks)

	modalities = {}
	for m in modes:
		results = solr.search(get_modality_query(brain_id=brain_id, modality=m))

		# split by section id
		results = split_by_section_id(results)

		# get first timestamp for each section id	
		results = get_first_timestamp(results)
	
	
		modalities[m] = results

	bf_masks, blockfaces, mod_masks  = filter_sections(bf_masks, blockfaces, mod_masks, modalities)

	modalities['BF-Mask'] = bf_masks
	modalities['Blockface'] = blockfaces
	modalities['TRANS-Mask'] = mod_masks
	
	assert_each_section_available_in_each_modality(modalities)
	for k in modalities:
		print(k, len(modalities[k]))

	with open('pli_big_brain_paths.json', 'w') as file:
		json.dump(modalities, file, indent=4)


if __name__ == "__main__":
	main()

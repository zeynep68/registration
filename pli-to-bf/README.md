# 3D reconstruction of blockface images

## Step 1

Run the following command:

```
python3 get_blockface_transformation_paths.py [--brain_id BRAIN_ID] [--path_to_transformations /path/to/transformation/dir/ [--solr_auth /path/to/yaml/file] [--output_file /path/to/output/file]
```


1. **brain_id**:  
   - `<PE-2021-00981-H> or <PE-2020-00691-H>` 

2. **path_to_transformations**: 
   - Use triple backticks ```` ``` ```` for code blocks. 
   - Specify the language after the opening triple backticks for syntax highlighting (e.g., `sh` for shell scripts).

3. **solr_auth**:
   - Path to the authentication yaml file.
   - Example YAML file content:
   ```yaml
   auth:
        user: user
        password: password 
   ```

4. **output_file**:
   - Path to the output file where the results will be saved.
   - Each line in the output_file contains 3 entries separated by ";"
   `section;path_to_blockface_img;path_to_transformation_matrix`

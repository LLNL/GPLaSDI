import yaml
import os
template_file = 'burgers1d-nonadaptive.yml'

with open(template_file, 'r') as file:
    template = yaml.safe_load(file)

# Modify variables lasdi: gplasdi: max_iter and latent space: ae: hidden_units:
# max_iters = [5000, 10000, 15000, 20000]
max_iters = [27800]
hidden_units = [25, 50, 100, 150]
for max_iter in max_iters:
    for hidden_unit in hidden_units:
        new_template = template.copy()
        new_template['lasdi']['gplasdi']['max_iter'] = max_iter
        new_template['lasdi']['gplasdi']['n_iter'] = max_iter
        new_template['latent_space']['ae']['hidden_units'] = [hidden_unit]
        exp_key = f"burgers1d-NA-MI{max_iter}-HU{hidden_unit}"
        new_template['exp_key'] = exp_key
        new_template['lasdi']['gplasdi']['path_checkpoint'] = f"{exp_key}/checkpoint"
        new_template['lasdi']['gplasdi']['results'] = f"{exp_key}"
        folder_name = exp_key
        # Create folder if it does not exist
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(folder_name + '/checkpoint/', exist_ok=True)
        with open(folder_name + '/config.yaml', 'w') as outfile:
            yaml.dump(new_template, outfile, default_flow_style=False)

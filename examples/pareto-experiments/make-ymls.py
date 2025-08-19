import yaml

template_file = 'burgers1d-nonadaptive.yml'

with open(template_file, 'r') as file:
    template = yaml.safe_load(file)

# Modify variables lasdi: gplasdi: max_iter and latent space: ae: hidden_units:
max_iters = [5000, 10000, 15000, 20000]
hidden_units = [25, 50, 100, 150]
for max_iter in max_iters:
    for hidden_unit in hidden_units:
        new_template = template.copy()
        new_template['lasdi']['gplasdi']['max_iter'] = max_iter
        new_template['latent_space']['ae']['hidden_units'] = [hidden_unit]
        exp_key = f"burgers1d-NA-MI{max_iter}-HU{hidden_unit}"
        new_template['exp_key'] = exp_key
        filename = f"burgers1d-NA-MI{max_iter}-HU{hidden_unit}.yml"
        with open(filename, 'w') as outfile:
            yaml.dump(new_template, outfile, default_flow_style=False)
        print(f"Created: {filename}")

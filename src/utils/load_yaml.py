from strictyaml import load

def load_yaml_to_pydict(path_yaml):

    with open(path_yaml, "r") as yml:
        obj_strict = load(yml.read())
        obj_dict = obj_strict.data

    return obj_dict  
import os

def rename_all():

    path = "trained_models"

    rename_map = {}

    for lang_dir in os.listdir(path):
        for old_model_name in os.listdir(path + "/" + lang_dir):
            ext = os.path.splitext(old_model_name)
            if ext[-1] == ".model":
                new_model_name = []
                parts = ext[0].split("_")

                if parts[0] != "ft":
                    new_model_name = ["w2v"]
                else:
                    parts[0] = "ft"

                new_model_name.extend(parts)
                new_model_name = new_model_name[:3]

                if "sg" in old_model_name:
                    new_model_name.append("sg")
                else:
                    new_model_name.append("cbow")

                new_model_name.append("st")
                new_model_name = "_".join(new_model_name)
                new_model_name += ".model"
                rename_map[old_model_name] = new_model_name

    for lang_dir in os.listdir(path):
        for model_name in os.listdir(path + "/" + lang_dir):
            old_model_name = model_name
            print("OLD", model_name)
            for key, val in rename_map.items():
                if key in model_name:
                    if "ft" not in model_name:
                        model_name = model_name.replace(key, val)
                    elif "ft" in val:
                        model_name = model_name.replace(key, val)

            print("NEW", model_name, "\n")
            old_path = path + "/" + lang_dir + "/" + old_model_name
            new_path = path + "/" + lang_dir + "/" + model_name

            os.rename(old_path, new_path)


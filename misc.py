from glob import glob
from tensorflow.keras.models import load_model, model_from_json

def get_model(training_idx: int = None, custom_objects: dict = None, compile = False, model_json_name: str = None, json_models_path: str = None):
    """
    
    """

    if training_idx != None:
        model_save_path = glob(f"logs/**/{training_idx}/model", recursive = True)

        if model_save_path.__len__() != 1:
            raise Exception(f"No training, or multiple trainings, found with the training index {training_idx}. This shouldn't be happening")

        model_save_path = model_save_path[0]

        model = load_model(model_save_path, custom_objects = custom_objects, compile = compile)

        return model

    if model_json_name:

        if not json_models_path:

            path = glob(f"**/{model_json_name}", recursive = True)

            if path.__len__() != 1:
                raise Exception(f"{path.__len__()} files, found with the name {model_json_name}: {path}")

            path = path[0]

        else:
            path = f"{json_models_path}/{model_json_name}"


        with open(path, 'r') as json_file:
                architecture = json_file.read()
                model = model_from_json(architecture)
                json_file.close()

        return model
import socket
from pprint import pprint

from tqdm import tqdm

from april import Dataset
from april.alignments.binet import BINet
from r2pa.discovery import discovery
from april.fs import MODEL_DIR, get_model_files
from r2pa.discovery.next_event_prediction.next_event_prediction import NextEventPredictor
from r2pa.discovery.process_model import ProcessModel
from scripts.discovery.training.evaluate_model_on_dataset import evaluate_models
from scripts.notifications import notify


def get_binet_model(dataset, model, use_present_activity, use_present_attributes):
    """ Loads the specified BINet model. """
    binet = BINet(dataset,
                  use_event_attributes=True,
                  use_case_attributes=False,
                  use_present_activity=use_present_activity,
                  use_present_attributes=use_present_attributes)
    binet([f[:1] for f in dataset.features])
    binet.load_weights(str(MODEL_DIR / model))
    return binet


def get_transformer_model(dataset, model, number_layers, number_heads):
    """ Loads the specified transformer model. """
    transformer = TransformerModel(dataset, num_encoder_layers=number_layers, mha_heads=number_heads)
    transformer.load(str(MODEL_DIR / model))
    return transformer


def get_transformer_configuration(dataset, model_file):
    ground_truth_model = ProcessModel(dataset=dataset)

    model = get_transformer_model(dataset, model_file, 6, 6)
    configuration = (dataset, model, ground_truth_model,
                     NextEventPredictor.from_transformer_with_cache,
                     NextEventPredictor.from_transformer,
                     discovery.create_detect_cache, "TR_6_6")

    return configuration


def get_binet_configuration(dataset, model):
    ground_truth_model = ProcessModel(dataset=dataset)
    use_present_activity, use_present_attributes = [bool(int(o)) for o in model.file.split('_')[1][-2:]]
    model = get_binet_model(dataset, model.file, use_present_activity, use_present_attributes)
    if use_present_activity and use_present_attributes:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_attribute_conditioned_keras_model_with_cache,
                NextEventPredictor.from_attribute_conditioned_keras_model,
                discovery.create_detect_cache_for_attribute_conditioning, "BINetV3")
    elif use_present_activity and not use_present_attributes:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_conditioned_keras_model_with_cache,
                NextEventPredictor.from_conditioned_keras_model,
                discovery.create_detect_cache_for_activity_conditioning, "BINetV2")
    else:
        return (dataset, model, ground_truth_model,
                NextEventPredictor.from_keras_model_with_cache,
                NextEventPredictor.from_keras_model,
                discovery.create_detect_cache, "BINetV1")


def model_evaluation():
    evaluation_configuration = []

    models = sorted(
        list(
            f for f in get_model_files()
            if f.model == 'paper'
            and f.model_name == 'TR'
            and f.id == 1
        ),
        key=lambda x: x.name
    )

    pprint([m.name for m in models])

    for model in tqdm(models, desc='Prepare'):
        dataset = Dataset(model.event_log_name, use_event_attributes=True)
        if model.model_name == "TR":
            evaluation_configuration.append(get_transformer_configuration(dataset, model.file))
        else:
            evaluation_configuration.append(get_binet_configuration(dataset, model))

    # evaluate all
    for configuration in tqdm(evaluation_configuration, desc='Evaluate'):
        evaluate_models(configuration)
        notify(f'Configuration `{str(configuration)}` done')


if __name__ == '__main__':
    try:
        model_evaluation()
        notify(f'`{__file__}` on `{socket.gethostname()}` has finished')
    except Exception as err:
        notify(f'`{__file__}` on `{socket.gethostname()}` has crashed with the following '
               f'message.\n\n```\n{err}\n```')
        raise err

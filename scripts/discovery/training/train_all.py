import socket

from scripts.discovery.training import train_binets
from scripts.notifications import notify

if __name__ == '__main__':
    # generate datasets
    # anomalies = [
    #     SkipSequenceAnomaly(max_sequence_size=2),
    #     ReworkAnomaly(max_distance=5, max_sequence_size=3),
    #     EarlyAnomaly(max_distance=5, max_sequence_size=2),
    #     LateAnomaly(max_distance=5, max_sequence_size=2),
    #     InsertAnomaly(max_inserts=2),
    #     AttributeAnomaly(max_events=3, max_attributes=2)
    # ]

    # process_models = [m for m in get_process_model_files() if 'testing' not in m]
    # for process_model in tqdm(process_models, desc='Generate'):
    #     generate_for_process_model(process_model,
    #                                size=5000,
    #                                anomalies=anomalies,
    #                                num_attr=[1, 2, 3, 4],
    #                                seed=1337)

    try:
        train_binets.train()
        # train_transformers.main()
        notify(f'`{__file__}` on `{socket.gethostname()}` has finished')
    except Exception as err:
        notify(f'`{__file__}` on `{socket.gethostname()}` has crashed with the following message.\n\n```\n{err}\n```')
        raise err

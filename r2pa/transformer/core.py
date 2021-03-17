import tensorflow as tf
import numpy as np

from april import Dataset
from april.alignments.binet import AnomalyDetectionResult
from april.enums import Base, Heuristic, Mode, Strategy, AttributeType, FeatureType
from r2pa.transformer.encoder import Encoder


class Transformer(tf.keras.Model):
    abbreviation = 'TR'
    name = 'Transformer'

    supported_bases = [Base.LEGACY, Base.SCORES]
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE, Mode.CLASSIFY]
    supports_attributes = True
    config = None
    version = None

    loaded = False

    def __init__(self,
                 dataset,
                 num_encoder_layers=3,
                 mha_heads=1,
                 dropout_rate=0.15,
                 ff_dim=64,
                 fixed_emb_dim=60,
                 use_present_attributes=False):
        super(Transformer, self).__init__()

        self.stop_training = False  # for early stopping

        # Parameters
        self.dataset = dataset

        # Layer lists
        self.inp = []
        self.pre_pre_outs = []
        self.pre_outs = []
        self.dropout_outs = []
        self.outs = []

        # This will also be the dim of the positional encoding; * 2 to guarantee an even number
        emb_dim = max([np.clip((dim + 1) // 20, 1, 5) * 2 for dim in dataset.attribute_dims]) \
            if fixed_emb_dim is None else fixed_emb_dim
        self.embedding_dimension = emb_dim

        inputs = zip(dataset.attribute_dims, dataset.attribute_keys, dataset.attribute_types, dataset.feature_types)
        for dim, key, t, feature_type in inputs:
            if feature_type != FeatureType.CASE:
                if t == AttributeType.CATEGORICAL:
                    voc_size = int(dim + 1)  # we start at 1, 0 is padding

                    embed = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True)
                    out = tf.keras.layers.Dense(voc_size, activation='softmax')
                else:
                    embed = tf.keras.layers.Dense(1, activation='linear')

                    out = tf.keras.layers.Dense(1, activation='linear')

                self.inp.append(embed)
                self.pre_pre_outs.append(tf.keras.layers.Dense(ff_dim, activation='relu'))
                self.pre_outs.append(tf.keras.layers.Dense(ff_dim, activation='relu'))
                self.dropout_outs.append(tf.keras.layers.Dropout(dropout_rate))
                self.outs.append(out)

        self.d_model = emb_dim
        self.use_present_attributes = use_present_attributes
        self.num_encoder_layers = num_encoder_layers
        self.number_mha_heads = mha_heads
        self.num_attributes = dataset.num_event_attributes
        self.max_case_length = dataset.max_len * self.num_attributes

        self.encoder = Encoder(num_encoder_layers, self.num_attributes, self.d_model, mha_heads, ff_dim,
                               self.max_case_length, rate=dropout_rate)

        self.look_ahead_mask = tf.linalg.band_part(
            tf.ones((self.max_case_length, self.max_case_length)), -1, 0)[self.num_attributes - 1::self.num_attributes]
        self.look_ahead_mask = tf.concat([self.look_ahead_mask[i:i + 1] for i in range(self.look_ahead_mask.shape[0])
                                          for _ in range(self.num_attributes)], axis=0)

        self.batch_size = 50

    def call(self, x_in, training=False,
             tape=None, track_embeddings=False, track_attention=False, return_input_embeddings=False):
        if not isinstance(x_in, list):
            x_in = [x_in]

        # Embedding
        x_emb = [input(x_, training=training) for x_, input in zip(x_in, self.inp)]
        emb_mask = x_emb[0]._keras_mask  # Mask is the same for all attributes
        x_emb = tf.transpose(x_emb, [1, 2, 0, 3])  # Reorder attributes to be consecutive per event
        x = tf.reshape(x_emb, [x_emb.shape[0], x_emb.shape[1] * x_emb.shape[2], x_emb.shape[3]])

        # Reapply Mask
        x._keras_mask = tf.concat([emb_mask[:, i:i + 1] for i in range(emb_mask.shape[1])
                                   for _ in range(self.num_attributes)], axis=-1)

        x, encoder_attentions, encoder_embeddings = self.encoder(x, training=training, mask=self.look_ahead_mask,
                                                                 tape=tape, track_attention=track_attention,
                                                                 track_embeddings=track_embeddings,
                                                                 return_input_embeddings=return_input_embeddings)

        x = tf.reshape(x, [x.shape[0], x.shape[1] // self.num_attributes, x.shape[2] * self.num_attributes])

        outputs = []
        for i, p_p, p, d, o in zip(range(self.num_attributes), self.pre_pre_outs, self.pre_outs, self.dropout_outs, self.outs):
            if self.use_present_attributes:
                present_attributes = tf.concat([x_emb[:, :, :i], x_emb[:, :, i + 1:]], axis=-2)
                present_attributes = tf.reshape(present_attributes,
                                                [*present_attributes.shape[:2],
                                                 present_attributes.shape[2] * present_attributes.shape[3]])

                x_local = tf.concat([x, present_attributes], axis=-1)
            else:
                x_local = x

            pre_out = d(p(p_p(x_local)), training=training)
            outputs.append(o(pre_out))

        return outputs, encoder_attentions, encoder_embeddings

    def score(self, features, predictions):
        # Add perfect prediction for start symbol

        for i, prediction in enumerate(predictions):
            p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            p[:, 0, features[i][0, 0]] = 1
            predictions[i] = p

        return transformer_scores_fn(np.dstack(features), predictions)

    def detect(self, dataset, batch_size=50, track_embeddings=False, track_attention=False, tape=None,
               return_input_embeddings=False):
        if isinstance(dataset, Dataset):
            # features = dataset.hierarchic_features
            features = dataset.features
            standard_features = dataset.features
        else:
            features = dataset
            standard_features = dataset

        # Get attentions and predictions
        predictions = [[] for _ in range(dataset.num_attributes)]
        attentions = [[[[] for _ in range(dataset.num_attributes)] for _ in range(dataset.num_attributes)]
                      for _ in range(self.num_encoder_layers)]
        embeddings = []

        for step in range(dataset.num_cases // batch_size):
            prediction, attentions_raw, embeddings_batch = \
                self([f[step * batch_size:(step + 1) * batch_size] for f in features], training=False,
                     track_attention=track_attention, track_embeddings=track_embeddings, tape=tape,
                     return_input_embeddings=return_input_embeddings)

            if track_embeddings:
                embeddings.append(tf.convert_to_tensor(embeddings_batch))

            for i in range(dataset.num_attributes):  # for prediction attribute
                predictions[i].append(prediction[i].numpy())

                if track_attention:
                    for k in range(len(attentions_raw)):  # encoder layer
                        # split into attention per attribute
                        outer_mask = tf.equal(tf.range(attentions_raw[k].shape[-2]) % dataset.num_attributes, i)
                        outer_masked = tf.boolean_mask(attentions_raw[k], outer_mask, axis=-2)

                        for j in range(dataset.num_attributes):
                            # encoding layer, prediction attribute (attended to), from attribute
                            attentions[k][i][j].append(outer_masked[j].numpy().astype(np.float16))

            if step % 10 == 0:
                print('prediction step %s' % step)

        if track_embeddings:
            embeddings = np.concatenate(embeddings, axis=1)

        for i in range(dataset.num_attributes):
            predictions[i] = np.concatenate(predictions[i], axis=0)

            if track_attention:
                for k in range(len(attentions)):
                    for j in range(dataset.num_attributes):
                        attentions[k][i][j] = np.concatenate(attentions[k][i][j], axis=0)

        if not isinstance(predictions, list):
            predictions = [predictions]

        return AnomalyDetectionResult(scores=[], predictions=predictions,
                                      attentions=attentions, attentions_raw=attentions_raw, embeddings=embeddings)

    def load(self, model_file):
        # TODO: Remove hardcoded values
        if not self.loaded:
            self.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy())
            self([f[:self.batch_size] for f in self.dataset.features])
            self.load_weights(str(model_file))
            self.loaded = True


def transformer_scores_fn(features, predictions):
    maxes = [np.repeat(np.expand_dims(np.max(p, axis=-1), axis=-1), p.shape[-1], axis=-1) for p in predictions]
    indices = [features[:, :, i:i + 1] for i in range(len(predictions))]
    scores_all = [(m - p) / m for p, m in zip(predictions, maxes)]

    scores = np.zeros(features.shape)
    for (i, j, k), f in np.ndenumerate(features):
        if f != 0 and k < len(scores_all):
            scores[i, j, k] = scores_all[k][i, j][indices[k][i, j]]

    return scores

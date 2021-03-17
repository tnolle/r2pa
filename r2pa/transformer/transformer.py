from r2pa.transformer.core import Transformer


class TransformerModel(Transformer):
    version = 1
    name, abbreviation = 'TR', 'Transformer'
    config = dict(dropout_rate=0.15, ff_dim=64, fixed_emb_dim=60)

    def __init__(self, dataset, **ad_kwargs):
        super(TransformerModel, self).__init__(dataset, **ad_kwargs, **self.config)

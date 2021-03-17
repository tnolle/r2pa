#  Copyright 2019 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================

import warnings

import arrow
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning

from april import Dataset, fs
from april.alignments.binet import BINet

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train():
    datasets = ['log2']

    for dataset_name in datasets:

        dataset = Dataset(dataset_name, use_event_attributes=True, use_case_attributes=False)
        x, y = dataset.features, dataset.targets

        number_attributes = dataset.num_attributes
        number_epochs = 1000
        batch_size = 10

        # for two attributes, v2 and v3 are the same
        #binet_versions = [(0, 0), (1, 0)]
        #if number_attributes > 2:
        #    binet_versions.append((1, 1))
        binet_versions = [(1, 1)]

        # for different versions of the binet
        for (present_activity, present_attribute) in binet_versions:
            start_time = arrow.now()

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)

            binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                          use_present_activity=present_activity, use_present_attributes=present_attribute)
            binet.fit(x=x, y=y, batch_size=batch_size, epochs=number_epochs, validation_split=0.2, callbacks=[callback])

            file = str(
                fs.MODEL_DIR / f'{dataset_name}_{binet.name}{present_activity}{present_attribute}_{start_time.format(fs.DATE_FORMAT)}')
            binet.save(file)
            binet.save_weights(f"{file}.h5")

            tf.keras.backend.clear_session()


if __name__ == '__main__':
    train()

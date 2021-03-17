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
#from scripts.notifications import notify


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train():
    datasets = ['paper-0.3-1']

    for dataset_name in datasets:
        for (present_activity, present_attribute, epochs) in [(False, False, 1000), (True, False, 1000)]:
            start_time = arrow.now()
            dataset = Dataset(dataset_name, use_event_attributes=True, use_case_attributes=False)
            binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                          use_present_activity=present_activity, use_present_attributes=present_attribute)
            x, y = dataset.features, dataset.targets
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            binet.fit(x=x, y=y, batch_size=500, epochs=epochs, validation_split=0.1, callbacks=[callback])
            binet.save(str(fs.MODEL_DIR / f'{dataset_name}_{binet.name}_{start_time.format(fs.DATE_FORMAT)}'))
            binet.save_weights(str(
                fs.MODEL_DIR / f'{dataset_name}_{binet.name}_{start_time.format(fs.DATE_FORMAT)}.h5'))

            tf.keras.backend.clear_session()


if __name__ == '__main__':
    train()

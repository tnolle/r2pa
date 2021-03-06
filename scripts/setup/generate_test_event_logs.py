# Copyright 2019 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

from tqdm import tqdm

from april.fs import get_process_model_files
from april.generation import *
from april.generation.utils import generate_for_process_model

anomalies = [
    SkipSequenceAnomaly(max_sequence_size=3),
    ReworkAnomaly(max_distance=5, max_sequence_size=5),
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=4),
    SkipAnomaly(max_skips=3),
    SkipAndInsertAnomaly(max_skips=3, max_inserts=4),
    ReplaceAnomaly(max_replacements=4),
    AttributeAnomaly(max_events=4, max_attributes=2)
]

process_models = [m for m in get_process_model_files()]
for process_model in tqdm(process_models, desc='Generate'):
    generate_for_process_model(process_model,
                               size=5000,
                               anomalies=anomalies,
                               num_attr=[0, 2, 4],
                               postfix='+test',
                               seed=42)

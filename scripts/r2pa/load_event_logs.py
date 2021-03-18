from april import Dataset
from april.fs import EVENTLOG_DIR
from april.processmining import EventLog


def convert_xes_to_json_event_log(file_name, remove_attributes, rename_attributes):
    """ Loads an event log stored in the XES format and saves it as a json file. """
    event_log = EventLog.from_xes(str(EVENTLOG_DIR / file_name), remove_attributes=remove_attributes,
                                  rename_attributes=rename_attributes)
    new_file_name = f"{file_name.split('.')[0]}.json.gz"
    event_log.save_json(EVENTLOG_DIR / new_file_name)


if __name__ == '__main__':
    convert_xes_to_json_event_log('log3.xes', remove_attributes=['Value', 'Row'],
                                  rename_attributes={})
    dataset = Dataset('log3', use_event_attributes=True,
                      attribute_order=['name', 'source', 'Url', 'Label', 'FileName', 'Sheet', 'Column'])
    test = 0

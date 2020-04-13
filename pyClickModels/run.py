import argparse
import json
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

# from pyClickModels.DBN import fuu

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    dest='input',
    default='/tmp/judgments_model_test_data.gz',
    help='Input file to process.')
parser.add_argument(
    '--output',
    dest='output',
    default='/tmp/output',
    help='Output file to write results to.')
known_args, pipeline_args = parser.parse_known_args([])

pipeline_options = PipelineOptions(pipeline_args)
pipeline_options.view_as(SetupOptions).save_main_session = True
p = beam.Pipeline(options=pipeline_options)

lines = p | 'read' >> ReadFromText(known_args.input)

d = {}


class Saved(beam.DoFn):
    def __init__(self, d):
        if 'hi' not in d:
            d['hi'] = 0
        self.d = d

    def process(self, element):
        self.d['hi'] += int(element) + 1
        print(self.d)


for _ in range(3):
    r = (
        lines
        | 'json read{_}' >> beam.Map(lambda x: json.loads(x))
        | f'get{_}' >> beam.Map(lambda x: int(x['judgment_keys'][0]['session'][0][
            'click']))
        | f'save d{_}' >> beam.ParDo(Saved(d))
    )


_ = r | 'write' >> WriteToText(known_args.output, file_name_suffix='.gz')


result = p.run()
result.wait_until_finish()

print(d)

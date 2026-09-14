import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

spec=importlib.util.spec_from_file_location('archive_surveys',Path(__file__).resolve().parents[1]/'sources/archive_surveys.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ArchiveSurveyTests(unittest.TestCase):
    def test_subtitles_keep_whole_document_times_and_unknown_speaker(self):
        raw='<document><s id="1"><time value="00:01"/>不是零，是001。<time value="00:02"/></s><s id="2">好，保留空值。</s><meta><id>work-1</id></meta></document>'.encode()
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:z.writestr('zh/1.xml',raw)
        archive.seek(0);archive.size=len(archive.getvalue())
        result=module.sample_zip(archive,{'path':'subtitles.zip','shard_probability':1},
            {'member_patterns':['*.xml'],'member_format':'xml-subtitles'},12,200,lambda _: {})
        row=result['samples'][0]
        self.assertEqual(row['text'],'不是零，是001。\n好，保留空值。')
        self.assertEqual(row['original']['segments'][0]['times'],[{'value':'00:01'},{'value':'00:02'}])
        self.assertEqual(row['original']['speaker_identity'],'unknown')
        self.assertFalse(row['original']['work_identity_verified'])
        self.assertEqual(row['inclusion_probability'],1)

    def test_whole_tables_preserve_raw_bytes_and_two_stage_probability(self):
        sink=io.BytesIO();pq.write_table(pa.table({'id':['001','002'],'n':[float('inf'),None]}),sink)
        raw=sink.getvalue(); archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:
            for i in range(4):z.writestr(f'{i}.parquet',raw)
            z.writestr('README.md','not a table')
        archive.seek(0);archive.size=len(archive.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            r=module.sample_zip(archive,{'path':'tables.zip','shard_probability':.5},
                                {'member_patterns':['*.parquet'],'member_format':'parquet-table'},12,2,lambda _: {},out)
            self.assertEqual(r['rows'],4)
            self.assertEqual(len(r['samples']),2)
            for row in r['samples']:
                self.assertEqual(row['inclusion_probability'],.25)
                self.assertEqual(row['original']['rows'][0]['id'],'001')
                self.assertIsNone(row['original']['rows'][1]['n'])
                self.assertEqual(row['original']['rows'][0]['n'],{'__mei_survey_nonfinite_float__':'inf'})
                self.assertEqual((out/row['raw_member_file']).read_bytes(),raw)
                self.assertEqual(row['member_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertIsNone(r['archive_sha256'])

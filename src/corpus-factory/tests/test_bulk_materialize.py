import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sources'))
import pyarrow as pa
import pyarrow.parquet as pq
from bulk_materialize import bulk, group_selection

class BulkTest(unittest.TestCase):
    def test_deterministic_groups_and_streamed_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'input.parquet'
            pq.write_table(pa.table({'text':['001\n字段','keep','keep','excluded','','final'],'id':list(range(6))}),source,row_group_size=2)
            meta=pq.ParquetFile(source).metadata
            self.assertEqual(group_selection(meta,.5,7,'a'),group_selection(meta,.5,7,'a'))
            exclusion=root/'exclude.jsonl'; exclusion.write_text(json.dumps({'text':'excluded'})+'\n')
            config={'mode':'parquet_bulk','source_id':'test','files':[{'path':str(source)}],
                    'row_group_fraction':1,'seed':7,'reserve_disk_bytes':0,
                    'exclude_jsonl':[str(exclusion)]}
            class Tokenizer:
                def encode_document(self,text):return [0]+list(text)+[1]
            with patch('bulk_materialize.load_tokenizer',return_value=Tokenizer()),patch('bulk_materialize.tokenizer_pointer',return_value={'test':True}):
                report=bulk(config,root/'out')
            rows=[json.loads(x) for x in (root/'out/part-0000.jsonl').read_text().splitlines()]
            self.assertEqual([r['text'] for r in rows],['001\n字段','keep','final'])
            self.assertEqual([r['origin']['row_index'] for r in rows],[0,1,5])
            self.assertEqual(report['records'],3)
            self.assertEqual(report['tokens'],sum(len(r['text'])+2 for r in rows))
            self.assertEqual(report['counts']['exact_or_excluded_duplicate'],2)
    def test_no_implicit_network(self):
        with self.assertRaises(ValueError):
            bulk({'files':[{'url':'https://example.invalid','path':'p'}]},Path('/unused'))

if __name__=='__main__':unittest.main()

import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

spec=importlib.util.spec_from_file_location('local_diagnostics',Path(__file__).resolve().parents[1]/'sources/local_diagnostics.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class ToolPreflightTests(unittest.TestCase):
    def test_subtitle_directory_and_upstream_labels_do_not_claim_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'survey/zh').mkdir(parents=True)
            sample={'archive_entries':[{'path':'zh/work/1.xml'},{'path':'zh/work/2.xml'},{'path':'INFO'}],
                'samples':[{'member':{'path':'zh/work/1.xml'},'text':'你好','original':{'metadata_xml':['<meta><subtitle><machine_translated>0</machine_translated></subtitle></meta>']}}]}
            path=root/'survey/zh/sample-one.json';path.write_text(json.dumps(sample))
            (root/'survey/survey.json').write_text(json.dumps({'artifacts':{'zh/sample-one.json':d.fingerprint(path)}}))
            result=d.subtitle_inventory(root,{'survey':'survey'});row=result['sources']['zh']
            self.assertEqual(row['frame_documents'],2)
            self.assertEqual(row['frame_extra_files_sharing_directory'],1)
            self.assertEqual(row['sample_metadata']['source/original'],{'unknown':1})
            self.assertEqual(row['sample_metadata']['subtitle/machine_translated'],{'0':1})
            self.assertNotIn('quality_passed',row)

    def test_declared_names_do_not_rewrite_strings_or_serialize_dependencies(self):
        value='[Read Table(text="Read Table(x=1), )", id="001"), Save(data=[{"a":true}])]'
        calls=d.parse_declared_tool_calls(value,['Read Table','Save'])
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['name'],'Read Table')
        self.assertEqual(calls[0]['arguments'],{'text':'Read Table(x=1), )','id':'001'})
        self.assertEqual(calls[1]['arguments'],{'data':[{'a':True}]})
        for bad in ('[Unknown()]','[Read Table(x=Save())]','[Save(),]','[Save()] trailing'):
            with self.assertRaises(ValueError):d.parse_declared_tool_calls(bad,['Read Table','Save'])

    def test_literals_preserve_identifiers_booleans_null_and_negative_values(self):
        calls=d.parse_literal_tool_calls('[check(id="001", value=-1.25, empty=null, flag=false, rows=[{"a":0}])]')
        self.assertEqual(calls,[{'name':'check','arguments':{'id':'001','value':-1.25,'empty':None,'flag':False,'rows':[{'a':0}]}}])
        for unsafe in ('[f(x=__import__("os").getcwd())]','[f(**a)]','[f(x=1,x=2)]','[f(x={"a":1,"a":2})]','[f(1)]','[a.f()]'):
            with self.assertRaises((ValueError,SyntaxError)):d.parse_literal_tool_calls(unsafe)

    def test_schema_type_alias_does_not_rewrite_literal_values(self):
        original={'type':'dict','properties':{'type':{'type':'string','default':'int','examples':[{'type':'int'}]},'n':{'type':'int'}}}
        result=d.map_schema_types(original)
        self.assertEqual(result['type'],'object')
        self.assertEqual(result['properties']['n']['type'],'integer')
        self.assertEqual(result['properties']['type'],original['properties']['type'])
        self.assertEqual(original['type'],'dict')

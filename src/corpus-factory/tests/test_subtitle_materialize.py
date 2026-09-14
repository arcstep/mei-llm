import io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sources'))
from subtitle_materialize import subtitle_bulk

class SubtitleTest(unittest.TestCase):
    def test_versions_are_grouped_but_ted_talks_are_independent(self):
        content=io.BytesIO()
        with zipfile.ZipFile(content,'w') as z:
            for name,text,flag in [('work/old.xml','旧版','1'),('work/new.xml','新版本','0'),('other/new.xml','另一作品','0')]:
                z.writestr(name,f'<document><s>{text}</s><meta><subtitle><machine_translated>{flag}</machine_translated></subtitle></meta></document>')
        raw=content.getvalue()
        class Response(io.BytesIO):status=200
        class Tokenizer:
            def encode_document(self,text):return [0]+list(text)+[1]
        with tempfile.TemporaryDirectory() as d,patch('subtitle_materialize.urllib.request.urlopen',side_effect=lambda *a,**k:Response(raw)),patch('subtitle_materialize.load_tokenizer',return_value=Tokenizer()),patch('subtitle_materialize.tokenizer_pointer',return_value={'test':True}):
            root=Path(d)
            report=subtitle_bulk({'sources':[{'source_id':'a','kind':'opensubtitles','url':'https://example.invalid/a','expected_bytes':len(raw)}]},root/'out',True)
            self.assertEqual(report['records'],2)
            self.assertEqual(report['outputs'][0]['counts']['declared_machine_translation'],1)
            from subtitle_materialize import work_key
            self.assertNotEqual(work_key('ted/a.xml','ted'),work_key('ted/b.xml','ted'))
            self.assertEqual(work_key('movie/a.xml','opensubtitles'),work_key('movie/b.xml','opensubtitles'))

if __name__=='__main__':unittest.main()

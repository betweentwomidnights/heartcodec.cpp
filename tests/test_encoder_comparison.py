"""Exercise exact-code and backend-profile gates without model inference."""
import hashlib,importlib.util,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('numpy'),'Offline encoder comparison requires NumPy')
class EncoderComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ref=self.root/'ref';self.native=self.root/'native'
        case=self.ref/'source-0';case.mkdir(parents=True);self.native.mkdir()
        self.model=self.root/'fixture.gguf';self.model.write_bytes(b'comparison-only model identity fixture')
        codes=struct.pack('<8i',*range(8));(case/'codes.i32').write_bytes(codes);(self.native/'codes.i32').write_bytes(codes)
        activation=struct.pack('<f',1.0);(case/'query.f32').write_bytes(activation)
        self.manifest=dict(heartlib_revision='fixture',encoder_revision='fixture',decoder_revision='fixture',device='cpu',source_complete_frames=1,
                           cases=[dict(name='source-0',files=dict(codes=dict(file='codes.i32',sha256=hashlib.sha256(codes).hexdigest()),query=dict(file='query.f32',sha256=hashlib.sha256(activation).hexdigest())))])
        self.run_data=dict(mode='encode',backend='cpu');self.save()
    def save(self):
        (self.ref/'manifest.json').write_text(json.dumps(self.manifest));(self.native/'run.json').write_text(json.dumps(self.run_data))
    def compare(self,*extra):
        report=self.root/'report.json'
        result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/compare-encoder.py'),'--reference',str(self.ref),'--native',str(self.native),'--model',str(self.model),'--output',str(report),*extra],capture_output=True,text=True)
        return result,json.loads(report.read_text()) if report.exists() else None
    def test_code_only_success_never_claims_activation_parity(self):
        result,report=self.compare('--mode','codes','--require-matching-backend')
        self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(report['code_parity_passed']);self.assertIsNone(report['all_stage_gates_passed'])
        self.assertIn('activations not evaluated',report['scope']);self.assertTrue(report['backend_match'])
    def test_wrong_backend_is_rejected_when_required(self):
        self.run_data['backend']='cuda';self.save();result,report=self.compare('--mode','codes','--require-matching-backend')
        self.assertNotEqual(result.returncode,0);self.assertIsNone(report);self.assertIn('match the reference device',result.stderr)
    def test_missing_backend_is_rejected_when_required(self):
        del self.run_data['backend'];self.save();result,report=self.compare('--mode','codes','--require-matching-backend')
        self.assertNotEqual(result.returncode,0);self.assertIsNone(report)
    def test_single_code_change_still_fails_matching_backend(self):
        (self.native/'codes.i32').write_bytes(struct.pack('<8i',*range(7),8191));result,report=self.compare('--mode','codes','--require-matching-backend')
        self.assertEqual(result.returncode,1);self.assertEqual(report['codes']['changed_codes'],1);self.assertFalse(report['passed'])
    def test_unchanged_codes_do_not_hide_original_tensor_failure(self):
        stage=self.native/'stages/chunk-000000';stage.mkdir(parents=True);(stage/'query.f32').write_bytes(struct.pack('<f',1.01))
        result,report=self.compare('--mode','raw');self.assertEqual(result.returncode,1);self.assertTrue(report['code_parity_passed']);self.assertFalse(report['all_stage_gates_passed'])
    def test_changed_golden_is_rejected(self):
        (self.ref/'source-0/codes.i32').write_bytes(struct.pack('<8i',*range(1,9)))
        result,report=self.compare('--mode','codes');self.assertNotEqual(result.returncode,0);self.assertIsNone(report);self.assertIn('Changed golden',result.stderr)

if __name__=='__main__':unittest.main()

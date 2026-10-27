"""Acceptance contracts for a source-only, locally generated experiment."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,unittest

PACKAGE=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(PACKAGE/'src'))

class SourceOnlyWorkflowTests(unittest.TestCase):
    def test_release_contains_no_datasets_weights_or_prediction_records(self):
        binary=[p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*')
                if p.is_file() and p.suffix.lower() in {'.npz','.pt','.pth','.csv'}
                and not any(part in {'runs','.venv'} for part in p.relative_to(PACKAGE).parts)]
        self.assertEqual(binary,[],'Public package must contain source only')

    def test_pipeline_cli_is_available(self):
        result=subprocess.run([sys.executable,str(PACKAGE/'run.py'),'pipeline','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--profile',result.stdout)
        self.assertIn('--workdir',result.stdout)

    def test_missing_checkpoint_never_falls_back_to_paper_weights(self):
        import reproduction as r
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):r.checkpoint_path('full',folder)

    def test_generated_split_seeds_are_disjoint(self):
        config=PACKAGE/'config/experiment.json'
        self.assertTrue(config.is_file(),'A complete source-only experiment configuration is required')
        settings=json.loads(config.read_text(encoding='utf-8'))
        seeds=settings['generation']['seeds']
        self.assertEqual(set(seeds),{'training','calibration','test'})
        self.assertEqual(len(set(seeds.values())),3)
        self.assertNotIn('original_training_indices',settings)

    def test_missing_run_directory_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,str(PACKAGE/'run.py'),'evaluate','--workdir',folder],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Run generate --workdir',result.stderr)

    def test_boundary_conditions_follow_the_active_saved_configuration(self):
        import reproduction as r
        self.assertTrue(callable(getattr(r,'boundary_conditions',None)),'Conditions must be derived when evaluating the saved run')
        old=r.PROTOCOL
        try:
            r.PROTOCOL={'main_loss_levels':[0.,.2],'receiver_error_levels':[0.,2.]}
            self.assertEqual(r.boundary_conditions(),[('random_loss',0.),('random_loss',.2),('receiver_error',0.),('receiver_error',2.)])
        finally:r.PROTOCOL=old

@unittest.skipUnless(os.environ.get('LDCR_TEST_WORKDIR'),'Set LDCR_TEST_WORKDIR to a completed local run for state-integrity tests')
class CompletedRunIntegrityTests(unittest.TestCase):
    def check_rejects_modified_metadata(self,relative,field):
        from workflow import check
        workdir=Path(os.environ['LDCR_TEST_WORKDIR'])
        self.assertTrue(check(workdir)['passed'],'The unmodified run must pass before a stale-metadata test')
        path=workdir/relative;original=path.read_text(encoding='utf-8')
        value=json.loads(original)
        if field=='checkpoints':value[field]['full']='stale-model-fingerprint'
        else:value[field]='stale-model-fingerprint'
        path.write_text(json.dumps(value),encoding='utf-8')
        try:
            with self.assertRaises(ValueError):check(workdir)
        finally:path.write_text(original,encoding='utf-8')

    def test_check_rejects_stale_calibration_encoder(self):
        self.check_rejects_modified_metadata('calibration/calibration_report.json','full_checkpoint_sha256')

    def test_check_rejects_stale_evaluation_weights(self):
        self.check_rejects_modified_metadata('results/evaluation_metadata.json','checkpoints')

    def test_check_rejects_modified_figure_even_with_valid_pdf_header(self):
        from workflow import check
        workdir=Path(os.environ['LDCR_TEST_WORKDIR'])
        self.assertTrue(check(workdir)['passed'])
        path=workdir/'figures/training_losses.pdf';original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n% stale-figure-test\n')
            with self.assertRaises(ValueError):check(workdir)
        finally:path.write_bytes(original)

if __name__=='__main__':unittest.main()

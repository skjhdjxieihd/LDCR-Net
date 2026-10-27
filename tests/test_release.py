"""Distribution-level checks for the anonymous source-only review snapshot."""
from pathlib import Path
import hashlib,json,re,unittest

ROOT=Path(__file__).resolve().parents[1]

class AnonymousReleaseTests(unittest.TestCase):
    def test_project_license_is_present(self):
        path=ROOT/'LICENSE'
        self.assertTrue(path.is_file(),'The release needs an explicit license')
        text=path.read_text(encoding='utf-8')
        self.assertTrue(text.startswith('MIT License'))
        self.assertIn('The project contributors',text)

    def test_upstream_notice_is_preserved(self):
        path=ROOT/'LICENSES/UnsupSeg-MIT.txt'
        self.assertTrue(path.is_file(),'CPC attribution requires the upstream permission notice')
        text=path.read_text(encoding='utf-8')
        self.assertIn('Copyright (c) 2020 Felix Kreuk',text)
        self.assertIn('Permission is hereby granted, free of charge',text)
        self.assertTrue((ROOT/'THIRD_PARTY_NOTICES.md').is_file())

    def test_introduction_and_provenance_are_anonymous(self):
        self.assertTrue((ROOT/'README.md').read_text(encoding='utf-8').startswith('# Anonymous review code'))
        data=json.loads((ROOT/'PROVENANCE.json').read_text(encoding='utf-8'))
        self.assertIn('components',data)
        self.assertNotIn('sources',data)
        self.assertNotIn('workflow_files',data)
        for component in data['components']:
            relative=Path(component['file'])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn('..',relative.parts)
            self.assertTrue((ROOT/relative).is_file())

    def test_no_emails_absolute_user_paths_or_repository_history(self):
        email=re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',re.I)
        windows=re.compile(r'(?<![A-Za-z])[A-Za-z]:[\\/]')
        homes=re.compile(r'/(?:home|Users)/[^/\s]+')
        for path in ROOT.rglob('*'):
            if not path.is_file() or any(p in {'runs','.venv','__pycache__'} for p in path.relative_to(ROOT).parts):continue
            self.assertNotIn('.git',path.relative_to(ROOT).parts)
            text=path.read_text(encoding='utf-8')
            self.assertIsNone(email.search(text),path.name)
            self.assertIsNone(windows.search(text),path.name)
            self.assertIsNone(homes.search(text),path.name)

if __name__=='__main__':unittest.main()

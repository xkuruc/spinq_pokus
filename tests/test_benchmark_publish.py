"""Check fast-forward result continuation with a disposable local Git remote."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import publish_results as publishing
from publish_results import publish_results


class ResultPublishingTests(unittest.TestCase):
    def test_resume_updates_branch_without_force_or_credential_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bare=root/"remote.git";repo=root/"repo";repo.mkdir()
            subprocess.run(["git","init","--bare",str(bare)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(["git","init",str(repo)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            archive=root/"archive.zip"
            with patch("publish_results._remote",return_value=str(bare)):
                archive.write_bytes(b"first")
                self.assertEqual(publish_results(repo,archive,"benchmark/test")["status"],"UPLOAD_SUCCEEDED")
                archive.write_bytes(b"continued")
                self.assertEqual(publish_results(repo,archive,"benchmark/test")["status"],"UPLOAD_SUCCEEDED")
                self.assertEqual(publish_results(repo,archive,"benchmark/test")["status"],"UPLOAD_SUCCEEDED")
                archive.write_bytes(b"accepted despite timeout")
                original_git=publishing._git
                def delayed_response(args,cwd,env):
                    response=original_git(args,cwd,env)
                    if "push" in args:
                        raise subprocess.TimeoutExpired(["git",*args],120)
                    return response
                with patch.object(publishing,"_git",side_effect=delayed_response):
                    self.assertEqual(publish_results(repo,archive,"benchmark/test")["status"],"UPLOAD_SUCCEEDED")
            content=subprocess.check_output(["git","--git-dir",str(bare),"show",
                                             "refs/heads/benchmark/test:results.zip"])
            self.assertEqual(content,b"accepted despite timeout")
            tree=subprocess.check_output(["git","--git-dir",str(bare),"ls-tree","--name-only",
                                          "refs/heads/benchmark/test"],text=True)
            self.assertEqual(tree.strip(),"results.zip")


if __name__=="__main__":unittest.main()

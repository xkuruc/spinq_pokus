"""Check compact result publication with a disposable local Git remote."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import publish_results as publishing
from publish_results import publish_summary


class ResultPublishingTests(unittest.TestCase):
    def test_only_final_analysis_is_published_and_timeout_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bare=root/"remote.git";repo=root/"repo";repo.mkdir()
            out=root/"results";out.mkdir();(out/"data").mkdir();(out/"plots").mkdir()
            subprocess.run(["git","init","--bare",str(bare)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(["git","init",str(repo)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            (out/"REPORT.md").write_text("final report\n",encoding="utf-8")
            (out/"comparison.csv").write_text("topic,error\nacquisition,1\n",encoding="utf-8")
            (out/"results.json").write_text(json.dumps({"state":"PILOT_COMPLETED_WITH_LIMITATIONS",
                "rows":[],"upload":{"status":"UPLOAD_FAILED"},"environment":{"host":"private"}}),
                encoding="utf-8")
            (out/"data"/"fid.json").write_bytes(b"raw chart marker")
            (out/"results.zip").write_bytes(b"full archive marker")
            (out/"plots"/"acquisition.png").write_bytes(b"plot marker")
            with patch("publish_results._remote",return_value=str(bare)):
                first=publish_summary(repo,out,"benchmark-summary/test")
                self.assertEqual(first["status"],"UPLOAD_SUCCEEDED")
                self.assertEqual(publish_summary(repo,out,"benchmark-summary/test")["status"],"UPLOAD_SUCCEEDED")
                (out/"REPORT.md").write_text("updated final report\n",encoding="utf-8")
                original_git=publishing._git
                def delayed_response(args,cwd,env):
                    response=original_git(args,cwd,env)
                    if "push" in args:
                        raise subprocess.TimeoutExpired(["git",*args],120)
                    return response
                with patch.object(publishing,"_git",side_effect=delayed_response):
                    self.assertEqual(publish_summary(repo,out,"benchmark-summary/test")["status"],"UPLOAD_SUCCEEDED")
            files=subprocess.check_output(["git","--git-dir",str(bare),"ls-tree","-r","--name-only",
                                           "refs/heads/benchmark-summary/test"],text=True).splitlines()
            self.assertEqual(set(files),{"README.md","REPORT.md","comparison.csv",
                                         "summary.json","plots/acquisition.png"})
            summary=json.loads(subprocess.check_output(["git","--git-dir",str(bare),"show",
                                    "refs/heads/benchmark-summary/test:summary.json"]))
            self.assertNotIn("upload",summary)
            self.assertNotIn("environment",summary)
            self.assertEqual(subprocess.check_output(["git","--git-dir",str(bare),"show",
                    "refs/heads/benchmark-summary/test:REPORT.md"]),b"updated final report\n")


if __name__=="__main__":unittest.main()

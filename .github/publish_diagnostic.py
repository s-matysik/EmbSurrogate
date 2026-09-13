"""Publish a CI diagnostic to the ci-diagnostics branch.

CI plumbing, not part of the package: GitHub stores job logs on rotating blob
hosts that a restricted network cannot reach, so each job writes its own file
through the contents API instead, where a plain API client can read it.

    python .github/publish_diagnostic.py <file> <tag>
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

path, tag = sys.argv[1], sys.argv[2]
token = os.environ["GH_TOKEN"]
repo = os.environ["GITHUB_REPOSITORY"]
url = f"https://api.github.com/repos/{repo}/contents/reports/{tag}.txt"
payload = {
    "message": f"ci diagnostic: {tag}",
    "branch": "ci-diagnostics",
    "content": base64.b64encode(open(path, "rb").read()).decode(),
}
for attempt in range(8):                      # four jobs race on the same branch
    try:
        r = urllib.request.Request(url, method="PUT",
                                   data=json.dumps(payload).encode(),
                                   headers={"Authorization": f"Bearer {token}",
                                            "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(r) as resp:
            print("published", resp.status)
            break
    except urllib.error.HTTPError as exc:
        print("retry", attempt, exc.code)
        time.sleep(4 + 4 * attempt)
else:
    print("could not publish the diagnostic")

import sys; sys.path.insert(0,".")
from trainforensics import remote
repos = sys.argv[1:]
for repo in repos:
    try:
        idx = remote.index(repo)
        names = [n for n in idx if "embed_tokens" in n or "lm_head" in n]
        out = []
        for n in names:
            try:
                out.append((n, remote.tensor_info(repo, n)))
            except Exception as e:
                out.append((n, "ERR " + repr(e)[:80]))
        print(repo, out)
    except Exception as e:
        print(repo, "ERR", repr(e)[:200])

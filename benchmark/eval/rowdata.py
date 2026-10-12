"""Row files of a result run, read from its data.tar.gz (see archive.sh).

local(path) extracts the archive next to it when a missing row file is asked for.
"""
import os
import tarfile

NAME = "data.tar.gz"


def local(path):
    if not os.path.exists(path):
        run = os.path.dirname(path) or "."
        archive = os.path.join(run, NAME)
        if os.path.exists(archive):
            with tarfile.open(archive) as t:
                try:
                    t.extractall(run, filter="data")
                except TypeError:
                    t.extractall(run)
    return path


if __name__ == "__main__":
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "src")
        os.mkdir(src)
        open(os.path.join(src, "a.tsv"), "w").write("x\n1\n")
        with tarfile.open(os.path.join(d, NAME), "w:gz") as t:
            t.add(os.path.join(src, "a.tsv"), "a.tsv")
        assert open(local(os.path.join(d, "a.tsv"))).read() == "x\n1\n"
        assert not os.path.exists(local(os.path.join(d, "b.tsv")))
        print("ok")
    finally:
        shutil.rmtree(d)

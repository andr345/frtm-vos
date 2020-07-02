from pathlib import Path
import os, sys
_srcdir = Path(__file__).resolve().parent
_build_dir = Path.home() / "tmp"
_build_dir.mkdir(parents=True, exist_ok=True)

from torch.utils.cpp_extension import load, verify_ninja_availability
try:
    verify_ninja_availability()
except:
    os.environ['PATH'] = str(Path(sys.executable).parent) + ":" + os.environ['PATH']

print("Compiling npp extension")
if (_build_dir / "lock").exists():
    print("Warning: found %s, compilation may hang here" % (_build_dir / "lock"))
nppig_cpp = load(verbose=False, name="nppig_cpp", sources=[_srcdir / "nppig.cpp"], extra_ldflags=['-lnppc', '-lnppig'],
                 with_cuda=True, build_directory=_build_dir)
print("done")

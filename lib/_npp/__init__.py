from pathlib import Path
import os, sys
_srcdir = Path(__file__).parent

from torch.utils.cpp_extension import load, verify_ninja_availability
try:
    verify_ninja_availability()
except:
    os.environ['PATH'] = str(Path(sys.executable).parent) + ":" + os.environ['PATH']

nppig_cpp = load(verbose=False, name="nppig_cpp", sources=[_srcdir / "nppig.cpp"], extra_ldflags=['-lnppc', '-lnppig'],
                 with_cuda=True, build_directory=Path.home() / "tmp")

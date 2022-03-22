
from __future__ import print_function

from setuptools import setup, find_packages, distutils
from torch.utils.cpp_extension import BuildExtension, CppExtension
import distutils.ccompiler
import distutils.command.clean
import glob
import inspect
import multiprocessing
import multiprocessing.pool
import os
import platform
import re
import shutil
import subprocess
import sys

base_dir = os.path.dirname('/opt/pytorch/pytorch')

def _get_build_mode():
    for i in range(1, len(sys.argv)):
        if not sys.argv[i].startswith('-'):
            return sys.argv[i]


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

def _compile_parallel(self,
                      sources,
                      output_dir=None,
                      macros=None,
                      include_dirs=None,
                      debug=0,
                      extra_preargs=None,
                      extra_postargs=None,
                      depends=None):
    # Those lines are copied from distutils.ccompiler.CCompiler directly.
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    list(
        multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()).imap(
            compile_one, objects))
    return objects


# Plant the parallel compile function.
if _check_env_flag('COMPILE_PARALLEL', default='1'):
    try:
        if (inspect.signature(distutils.ccompiler.CCompiler.compile) ==
                inspect.signature(_compile_parallel)):
            distutils.ccompiler.CCompiler.compile = _compile_parallel
    except BaseException:
        pass


class Clean(distutils.command.clean.clean):

    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


class Build(BuildExtension):

    def run(self):
        # Run the original BuildExtension first. We need this before building
        # the tests.
        BuildExtension.run(self)



build_mode = _get_build_mode()

# Fetch the sources to be built.
torch_lazy_sources = (glob.glob('csrc/*.cpp'))

# Constant known variables used throughout this file.
pytorch_source_path = os.getenv('PYTORCH_SOURCE_PATH',
                                os.path.dirname(base_dir))

# Setup include directories folders.
include_dirs = [
    base_dir,
    pytorch_source_path,
    os.path.join(pytorch_source_path, 'torch/csrc'),
    os.path.join(pytorch_source_path, 'torch/lib/tmp_install/include'),
]

extra_link_args = []

DEBUG = _check_env_flag('DEBUG')

extra_compile_args = [
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-unknown-pragmas',
    '-Wno-return-type',
]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']
else:
    extra_compile_args += ['-DNDEBUG']

setup(
    name=os.environ.get('TORCH_LAZY_PACKAGE_NAME', 'lazy_mode'),
    version=0.5,
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    ext_modules=[
        CppExtension(
            '_lazy',
            torch_lazy_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            #library_dirs=library_dirs,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': Build,
        'clean': Clean,
    })

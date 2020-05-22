from __future__ import absolute_import

import os
import sys
import Cython.Compiler.Options
from codecs import open
from Cython.Distutils import build_ext
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools.command.test import test as TestCommand


here = os.path.abspath(os.path.dirname(__file__))
Cython.Compiler.Options.annotate = True

_version = {}
_version_path = os.path.join(here, 'pyClickModels', '__version__.py')

with open(_version_path, 'r', 'utf-8') as f:
    exec(f.read(), _version)

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()


if sys.argv[-1] == 'publish':
    """
    Deploy to PyPI is still somewhat manual. It runs locally on Docker instead of relying
    on Travis.
    """
    os.system('./scripts/build_wheels.sh')
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload -r pypitest  dist/*')
    sys.exit()


def build_define_macros():
    """
    Macro CYTHON_TRACE is set to True so coverage report is available. More info in:

    https://stackoverflow.com/questions/50967268/cython-generating-coverage-for-pyx-file
    """
    args_ = sys.argv
    if len(args_) > 1:
        command = args_[1]
        define_macros = [('CYTHON_TRACE', '1')] if command == 'test' else []
    return define_macros


define_macros = build_define_macros()


class PyTest(TestCommand):

    user_options = [
        ('coverage=', None, 'Runs coverage report.'),
        ('html=', None, 'Saves result to html report.'),
    ]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []
        self.coverage = False
        self.html = False

    def finalize_options(self):
        TestCommand.finalize_options(self)

        if self.coverage:
            self.pytest_args.extend(['--cov-config', '.coveragerc'])
            self.pytest_args.extend([
                '--cov', 'pyClickModels', '--cov-report', 'term-missing'])

        if self.html:
            self.pytest_args.extend(['--cov-report', 'html'])

        self.pytest_args.extend(['-p', 'no:warnings'])

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


ext_modules = [
    Extension(
        'pyClickModels.DBN',
        ['pyClickModels/DBN.pyx'],
        language='c++',
        libraries=['json-c'],
        include_dirs=['pyClickModels'],
        define_macros=define_macros,
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"]
    ),
    Extension(
        'tests.test_cy_DBN',
        ['tests/test_cy_DBN.pyx'],
        language='c++',
        libraries=['json-c'],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"]
    )
]

install_requires = [
    'cython',
    'numpy',
    'ujson'
]

tests_require = [
    'pytest',
    'pytest-cov',
    'mock'
]

setup_requires = [
    'flake8',
    'isort',
    'pytest-runner'
]

extras_require = {
    'testing': tests_require
}

compiler_directives = {
    'language_level': '3',
    'binding': False,
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'linetrace': True
}

packages = ['pyClickModels']

setup(
    name='pyClickModels',
    version=_version['__version__'],
    author='Willian Fuks',
    author_email='willian.fuks@gmail.com',
    description='ClickModels for Search Engines Implemented on top of Cython.',
    packages=packages,
    include_package_data=True,
    package_data={
        'pyClickModels': ['*.pxd']
    },
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    license='MIT',
    ext_modules=cythonize(
        ext_modules,
        compiler_directives=compiler_directives
    ),
    cmdclass={
        'build_ext': build_ext,
        'test': PyTest
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering',
    ],
)

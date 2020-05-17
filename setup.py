import Cython.Compiler.Options
from Cython.Distutils import build_ext
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension


Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension(
        'pyClickModels.DBN2',
        ['pyClickModels/DBN2.pyx'],
        language='c++',
        libraries=['json-c'],
        include_dirs=['pyClickModels']
    ),
    Extension(
        'tests.test_cy_DBN2',
        ['tests/test_cy_DBN2.pyx'],
        language='c++',
        libraries=['json-c']
    )
]

install_requires = [
    'cython'
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
    'cdivision': True
}


packages = ['pyClickModels']

setup(
    name='pyClickModels',
    version='0.0.0',
    author='Willian Fuks',
    author_email='willian.fuks@gmail.com',
    packages=packages,
    include_package_data=True,
    package_data={'pyClickModels': ['*.pxd']},
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    license='MIT',
    ext_modules=cythonize(ext_modules, compiler_directives=compiler_directives),
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)

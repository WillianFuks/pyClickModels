.PHONY: flake8 coverage coverage-html test publish

flake8:
	pip install -U flake8
	flake8 pyClickModels

isort:
	pip install -U isort
	isort -rc pyClickModels
	isort -rc tests

isort-check:
	pip install -U isort
	isort -ns __init__.py -rc -c -df -p pyClickModels pyClickModels tests

coverage:
	python setup.py test --coverage=true

coverage-html:
	python setup.py test --coverage=true --html=true

test:
	python setup.py test

publish:
	pip install -U setuptools
	pip install -U wheel
	pip install 'twine>=1.5.0'
	pip install auditwheel
	sh ./scripts/build_wheels.sh
	#twine upload --repository testpypi dist/*
	twine upload dist/*
	#rm -fr build dist .egg *.egg-info

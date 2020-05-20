.PHONY: flake8 coverage coverage-html test

flake8:
	pip install -U flake8
	flake8

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

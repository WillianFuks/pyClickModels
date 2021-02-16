docker run -v $(pwd):/pyClickModels quay.io/pypa/manylinux1_x86_64 sh -c '''
yum update
yum install -y json-c-devel

cd /pyClickModels

for PYVER in /opt/python/*/bin/; do
    if [[ $PYVER != *"27"* ]]; then
        "${PYVER}/pip" install -U setuptools
        "${PYVER}/pip" install -r requirements.txt
        "${PYVER}/python" setup.py sdist bdist_wheel
    fi
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat "manylinux2010_x86_64" -w dist/
    rm $whl
done
'''

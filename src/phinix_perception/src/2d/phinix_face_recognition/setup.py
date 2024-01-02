from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'phinix_face_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'param'), glob('param/*.*')),
        (os.path.join('share', package_name, 'models'), glob('models/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jk',
    maintainer_email='jagadishkmahendran@gmail.com',
    description='PHINIX face recognition with optimized model to run on OpenVINO GPU',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "phinix_face_rec_py_exe = phinix_face_recognition.phinix_face_recognition:main",
            "phinix_face_reg_py_exe = phinix_face_recognition.phinix_face_registration:main"
        ],
    },
)

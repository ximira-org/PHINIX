import os
from glob import glob
from setuptools import setup

package_name = 'phinix_haptics_ui'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'param'), glob('param/*.*')),
        (os.path.join('share', package_name, 'settings'), glob('settings/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jk',
    maintainer_email='jagadishkmahendran@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "phinix_haptics_ui_py_exe = phinix_haptics_ui.phinix_haptics_ui:main",
            "phinix_haptics_obs_det_py_exe = phinix_haptics_ui.obstacle_det_haptics:main"
        ],
    },
)
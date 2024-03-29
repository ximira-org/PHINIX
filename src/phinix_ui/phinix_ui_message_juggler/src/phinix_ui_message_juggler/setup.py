import os
from glob import glob
from setuptools import setup

package_name = 'phinix_ui_message_juggler'

setup(
    name=package_name,
    version='0.0.0',
    # This is where you add packages
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'param'), glob('param/*.*')),
        (os.path.join('share', package_name, 'text_samples'), glob('text_samples/*.txt')),
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
            "phinix_ui_message_juggler_py_exe = phinix_ui_message_juggler.phinix_ui_message_juggler:main",
            "phinix_ui_juggler_simulator_py_exe = phinix_ui_message_juggler.phinix_ui_juggler_simulator:main",

        ],
    },
)

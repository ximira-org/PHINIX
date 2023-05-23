import os 
from glob import glob
from setuptools import setup

package_name = 'nova_tts_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "nova_tts_simulator_py_exe = nova_tts_simulator.nova_tts_simulator:main"
        ],
    },
)

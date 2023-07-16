from setuptools import setup

package_name = 'depthmap_gpu'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amrelsersy',
    maintainer_email='amrelsersay@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "depthmap = depthmap_gpu.depthmap:main",
            "publish_depthmaps = depthmap_gpu.publish_depthmaps:main"
        ],
    },
)

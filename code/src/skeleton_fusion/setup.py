from setuptools import find_packages, setup

package_name = 'skeleton_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='antoniocanetebaena1234@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_extractor_node = skeleton_fusion.pose_extractor_node:main',
            'skeleton_fusion_node = skeleton_fusion.skeleton_fusion_node:main',
            
        ],
    },
)

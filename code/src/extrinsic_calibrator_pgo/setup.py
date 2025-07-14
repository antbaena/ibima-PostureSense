from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'extrinsic_calibrator_pgo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='antoniocanetebaena1234@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = extrinsic_calibrator_pgo.camera_node:main',  
            'gtsam_aruco_detector_node = extrinsic_calibrator_pgo.gtsam_aruco_detector_node:main',
            'gtsam_camera_optimizer_node = extrinsic_calibrator_pgo.gtsam_camera_optimizer_node:main',
            'static_tf_broadcaster_node = extrinsic_calibrator_pgo.static_tf_broadcaster_node:main',
        ],
    },
)

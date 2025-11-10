from setuptools import find_packages, setup
import os
import glob
package_name = 'multicam_cube_calib'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),

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
            'extrinsics_optimizer_node = multicam_cube_calib.extrinsics_optimizer_node:main',
            'synchronized_pair_ICP_publisher_node = multicam_cube_calib.synchronized_pair_ICP_publisher_node:main',
            'synchronized_pair_PnP_publisher_node = multicam_cube_calib.synchronized_pair_PnP_publisher_node:main',
            'depth_to_pointcloud_plotter = multicam_cube_calib.depth_to_pointcloud_plotter:main',
            'pose_viz_node = multicam_cube_calib.pose_viz_node:main',
        ],
    },
)

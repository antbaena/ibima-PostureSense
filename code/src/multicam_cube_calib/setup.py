from setuptools import find_packages, setup

package_name = 'multicam_cube_calib'

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
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pair_builder_node = multicam_cube_calib.pair_builder_node:main',
            'pair_builder_test_node = multicam_cube_calib.pair_builder_test_node:main',
            'extrinsics_optimizer_node = multicam_cube_calib.extrinsics_optimizer_node:main',
            'cube_detector_node = multicam_cube_calib.cube_detector_node:main',
            'cube_detector_unsync_node = multicam_cube_calib.cube_detector_unsync_node:main',
            'calibration_manager_node = multicam_cube_calib.calibration_manager_node:main',
        ],
    },
)

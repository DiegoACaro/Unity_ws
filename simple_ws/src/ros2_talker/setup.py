from setuptools import find_packages, setup

package_name = 'ros2_talker'

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
    maintainer='juan',
    maintainer_email='juan_p.rivera@uao.edu.co',
    description='ROS 2 Talker Example',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_talker.talker:main',
            'listener = ros2_talker.listener:main',
            'boneRot = ros2_talker.boneRot:main',
        ],
    },
)

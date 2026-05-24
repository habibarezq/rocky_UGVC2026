from setuptools import find_packages, setup

package_name = 'rocky_perception'

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
    maintainer='abood',
    maintainer_email='anaalbakatoshy.gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "lane_filter_node = rocky_perception.lane_filter_node:main",
            "lane_follower_node = rocky_perception.lane_follower_node:main",
        ],
    },
)

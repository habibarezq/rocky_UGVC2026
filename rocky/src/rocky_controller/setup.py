from setuptools import find_packages, setup

package_name = 'rocky_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/controller.launch.py', 'launch/joystick.launch.py']),
        ('share/' + package_name + '/config', ['config/joystick.yaml']),
        ('share/' + package_name + '/config', ['config/my_controllers.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='habibarezq',
    maintainer_email='habibarezq30@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "twist_stamper = rocky_controller.twist_stamper:main",
            "noisy_controller = rocky_controller.noisy_controller:main",
        ],
    },
)

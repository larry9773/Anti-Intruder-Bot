from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'moving'

# Iterate through all the files and subdirectories to build the data files array
# We need this to maintain the file structure of `models/` in the installed package
def generate_data_files(share_path, folder):
    data_files = []

    for path, _, files in os.walk(folder):
        list_entry = (
            os.path.join(share_path, path),
            [os.path.join(path, f) for f in files if not f.startswith(".")],
        )
        data_files.append(list_entry)

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
(os.path.join("share", package_name, "worlds"), glob(os.path.join("worlds", "*"))),
    ] + generate_data_files(os.path.join("share", package_name), "models"),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ezy21kl',
    maintainer_email='ezy21kl@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'moving = moving.moving:main'
        ],
    },
)

from setuptools import setup

package_name = "person_follower"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/follower.launch.py"]),
        ("share/" + package_name + "/config", ["config/config.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LGP",
    maintainer_email="lgp@example.com",
    description="RoboMaster S1 YOLO11 person follower with Flask dashboard.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "person_follower_node = person_follower.person_follower_node:main",
        ],
    },
)

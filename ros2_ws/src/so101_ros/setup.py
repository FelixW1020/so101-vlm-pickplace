from setuptools import find_packages, setup

package_name = "so101_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felix Wang",
    maintainer_email="fw124@duke.edu",
    description="SO-101 two-stage language-grounded pick-and-place nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge_node = so101_ros.bridge_node:main",
            "grounding_node = so101_ros.grounding_node:main",
            "policy_node = so101_ros.policy_node:main",
            "orchestrator_node = so101_ros.orchestrator_node:main",
        ],
    },
)

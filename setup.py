from setuptools import setup

setup(
    name="kanake_mujoco",
    version="0.1.0",
    packages=["kanake_mujoco"],
    package_data={"kanake_mujoco": ["assets/urdf/*.urdf", "assets/meshes/*.STL"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/kanake_mujoco"]),
        ("share/kanake_mujoco", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "mujoco>=3.2.5,<4", "numpy"],
    python_requires=">=3.10",
    zip_safe=False,
    entry_points={"console_scripts": ["view = kanake_mujoco.view:main"]},
)

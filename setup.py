from setuptools import setup, find_packages

setup(
    name="ShurikodeContext",
    author="Riccardo Carraro",
    version="0.1.0",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    package_data={
        "shurikodecontext.ml_model": ["*.pth.tar"],
    },
    install_requires=["Pillow", "numpy", "torch", "torchvision"],
)

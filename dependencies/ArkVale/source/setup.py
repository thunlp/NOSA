from setuptools import setup, find_packages

setup(
    name="arkvale",
    packages=find_packages(),
    package_data={
        "arkvale": ["arkvale_cpp*.so"],
    },
    include_package_data=True,
    zip_safe=False,
)
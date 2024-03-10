from setuptools import find_packages, setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("patch_conv/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="patch_conv",
        author="Muyang Li, Ligeng Zhu, and Tianle Cai",
        author_email="muyangli@mit.edu",
        packages=find_packages(),
        install_requires=["torch"],
        url="https://github.com/mit-han-lab/patch_conv",
        description="Patch convolution to avoid large GPU memory usage of Conv2D",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )

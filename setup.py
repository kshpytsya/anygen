from setuptools import setup, find_packages

setup(
    name="anygen",
    description="",
    long_description=open("README.md").read(),  # no "with..." will do for setup.py
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    license="MIT",
    author="Kyrylo Shpytsya",
    author_email="kshpitsa@gmail.com",
    url="https://github.com/kshpytsya/anygen",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    install_requires=[
        'yaql>=1.1.3,<2',
        'jinja>=2.10,<3',
        'ruamel.yaml>=0.15,<1'
    ],
    python_requires=">=3.6, <=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)

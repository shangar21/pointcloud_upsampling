from setuptools import setup, find_packages

setup(
        name="bolt",
        version="0.1",
        packages=find_packages(include=['bolt', 'bolt.*']),
        install_requires=[
            'open3d',
            'numpy',
            'tqdm',
            'matplotlib'
        ]
)

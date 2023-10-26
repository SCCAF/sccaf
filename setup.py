from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


print(find_packages())

setup(
        name='SCCAF',
        version='0.0.11',
        description='Single-Cell Clustering Assessment Framework',
        long_description=readme(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'louvain',
            'scikit-learn',
            'psutil',
            'scanpy==1.9.2'],
        scripts=['cli/sccaf', 'cli/sccaf-assess', 'cli/sccaf-assess-merger', 'cli/sccaf-regress-out'],
        author='Chichau Miau',
        author_email='zmiao@ebi.ac.uk',
        license='MIT'
    )
# modified by Yuyao Song <ysong@ebi.ac.uk> on Oct 2023 to bump scanpy version

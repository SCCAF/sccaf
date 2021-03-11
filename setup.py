from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


print(find_packages())

setup(
        name='SCCAF',
        version='0.0.10',
        description='Single-Cell Clustering Assessment Framework',
        long_description=readme(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'louvain==0.6.1',
            'scikit-learn',
            'psutil',
            'scanpy==1.4.4'],
        scripts=['cli/sccaf', 'cli/sccaf-assess', 'cli/sccaf-assess-merger', 'cli/sccaf-regress-out'],
        author='Chichau Miau',
        author_email='zmiao@ebi.ac.uk',
        license='MIT'
    )

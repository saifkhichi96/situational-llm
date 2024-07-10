from setuptools import setup, find_packages

# read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='llm-cookbook',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'finetune-lora = recipes.finetune_lora:main',
            'chat = recipes.chat:main',
        ],
    },
)

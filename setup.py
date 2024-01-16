from setuptools import setup, find_packages

# Setting up
setup(
    name="SchemaLLM",
    version="0.0.0",
    author="AI-UI GmbH",
    author_email="info@ai-ui.ai",
    description="",
    long_description="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pydantic < 2.0",
    ],
    keywords=[],
    classifiers=[],
)

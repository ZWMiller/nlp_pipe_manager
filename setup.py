import setuptools

with open("README.md") as fh:
  README_CONTENTS = fh.read()

config = {
  'name': 'NLPPipeManager',
  'version': '0.1',
  'author': 'ZWMiller',
  'author_email': 'zach@notarealemail.com',
  'long_description': README_CONTENTS,
  'url': 'https://github.com/zwmiller/nlp_pipe_manager/',
  'packages': setuptools.find_packages()
}

setuptools.setup(**config)

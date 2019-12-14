from setuptools import setup

NAME = "paloboost"
VERSION = "0.0.1"
DESCR = "TBD"
URL = "TBD"
REQUIRES = []

AUTHOR = "Yubin Park"
EMAIL = "yubin.park@gmail.com"

LICENSE = "Apache 2.0"

SRC_DIR = "paloboost"
PACKAGES = [SRC_DIR]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE
          )


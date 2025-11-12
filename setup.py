from pkg_resources import parse_version
import setuptools
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# Settings merged from settings.ini
lib_name = 'box21_api'
version = '0.0.18'
description = 'Api for Box21.ai'
keywords = 'nbdev jupyter notebook python'
author = 'Ramon'
author_email = 'ramonhollands@gmail.com'
user = 'ramonhollands'
branch = 'main'
license_name = 'apache2'
status = '3'
min_python = '3.7'
audience = 'Developers'
language = 'English'
git_url = 'https://github.com/ramonhollands/box21_api'
lib_path = 'box21_api'

# Requirements
requirements = ['fastcore', 'pandas', 'pillow', 'requests', 'load_dotenv', 'tqdm']

# License mapping
licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
    'mit': ('MIT License', 'OSI Approved :: MIT License'),
    'gpl2': ('GNU General Public License v2', 'OSI Approved :: GNU General Public License v2 (GPLv2)'),
    'gpl3': ('GNU General Public License v3', 'OSI Approved :: GNU General Public License v3 (GPLv3)'),
    'bsd3': ('BSD License', 'OSI Approved :: BSD License'),
}

# Status mapping
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]

# Python versions
py_versions = '3.6 3.7 3.8 3.9 3.10'.split()

# License setup
lic = licenses.get(license_name.lower(), (license_name, None))

setuptools.setup(
    name = lib_name,
    version = version,
    description = description,
    keywords = keywords,
    author = author,
    author_email = author_email,
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(status)],
        'Intended Audience :: ' + audience.title(),
        'Natural Language :: ' + language.title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]] + (['License :: ' + lic[1] ] if lic[1] else []),
    url = git_url,
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    python_requires  = '>=' + min_python,
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = {
        "console_scripts": [
            "box21_export=box21_api.box21_export:main",
            "box21_import=box21_api.box21_import:main",
            "box21_export_preds=box21_api.box21_export_preds:main",
            "clean_annotations=box21_api.clean_annotations:main",
        ]
    },
)



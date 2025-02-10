import os
import sys
from os import path
import tempfile

homedir = path.expanduser("~")
tmpdir = tempfile.gettempdir()

def macos(name, subdirs:list[str] = []):
	library = path.join(homedir, 'Library')
	return {
		'data': path.join(library, 'Application Support', name, *subdirs),
		'config': path.join(library, 'Preferences', name, *subdirs),
		'cache': path.join(library, 'Caches', name, *subdirs),
		'log': path.join(library, 'Logs', name, *subdirs),
		'temp': path.join(tmpdir, name, *subdirs),
	}

def windows(name, subdirs:list[str] = []):
	app_data = os.environ.get('APPDATA') or path.join(homedir, 'AppData', 'Roaming')
	local_app_data = os.environ.get('LOCALAPPDATA') or path.join(homedir, 'AppData', 'Local')
	return {
		'data': path.join(local_app_data, name,"Data", *subdirs),
		'config': path.join(app_data, name, 'Config', *subdirs),
		'cache': path.join(local_app_data, name, 'Cache', *subdirs),
		'log': path.join(local_app_data, name, 'Log', *subdirs),
		'temp': path.join(tmpdir, name, *subdirs),
	}

def linux(name, subdirs:list[str] = []):
	username = path.basename(homedir)
	return {
		'data': path.join(os.environ.get('XDG_DATA_HOME') or path.join(homedir, '.local', 'share'), name, *subdirs),
		'config': path.join(os.environ.get('XDG_CONFIG_HOME') or path.join(homedir, '.config'), name, *subdirs),
		'cache': path.join(os.environ.get('XDG_CACHE_HOME') or path.join(homedir, '.cache'), name, *subdirs),
		'log': path.join(os.environ.get('XDG_STATE_HOME') or path.join(homedir, '.local', 'state'), name, *subdirs),
		'temp': path.join(tmpdir, username, name),
	}

def env_paths(name, subdirs:list[str] = []):
	if not isinstance(name, str):
		raise TypeError(f'Expected a string, got {type(name).__name__}')
	
	
	if sys.platform == 'darwin':
		return macos(name, subdirs)
	elif sys.platform == 'win32':
		return windows(name, subdirs)
	else:
		return linux(name, subdirs)
	

def get_data_path_for(subdirs:list[str]):
	package_name = "donew"
	return env_paths(package_name, subdirs)['data']



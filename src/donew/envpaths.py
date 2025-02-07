import os
import sys
from os import path
import tempfile

homedir = path.expanduser("~")
tmpdir = tempfile.gettempdir()

def macos(name):
	library = path.join(homedir, 'Library')
	return {
		'data': path.join(library, 'Application Support', name),
		'config': path.join(library, 'Preferences', name),
		'cache': path.join(library, 'Caches', name),
		'log': path.join(library, 'Logs', name),
		'temp': path.join(tmpdir, name),
	}

def windows(name):
	app_data = os.environ.get('APPDATA') or path.join(homedir, 'AppData', 'Roaming')
	local_app_data = os.environ.get('LOCALAPPDATA') or path.join(homedir, 'AppData', 'Local')
	return {
		'data': path.join(local_app_data, name, 'Data'),
		'config': path.join(app_data, name, 'Config'),
		'cache': path.join(local_app_data, name, 'Cache'),
		'log': path.join(local_app_data, name, 'Log'),
		'temp': path.join(tmpdir, name),
	}

def linux(name):
	username = path.basename(homedir)
	return {
		'data': path.join(os.environ.get('XDG_DATA_HOME') or path.join(homedir, '.local', 'share'), name),
		'config': path.join(os.environ.get('XDG_CONFIG_HOME') or path.join(homedir, '.config'), name),
		'cache': path.join(os.environ.get('XDG_CACHE_HOME') or path.join(homedir, '.cache'), name),
		'log': path.join(os.environ.get('XDG_STATE_HOME') or path.join(homedir, '.local', 'state'), name),
		'temp': path.join(tmpdir, username, name),
	}

def env_paths(name, suffix='python'):
	if not isinstance(name, str):
		raise TypeError(f'Expected a string, got {type(name).__name__}')
	
	if suffix:
		# Add suffix to prevent possible conflict with native apps
		name = f"{name}-{suffix}"
	
	if sys.platform == 'darwin':
		return macos(name)
	elif sys.platform == 'win32':
		return windows(name)
	else:
		return linux(name)
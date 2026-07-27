"""
Runtime hook to fix module paths when running from PyInstaller EXE.
This ensures viphoneme and other packages can find their data correctly.
"""
import sys
import os

# When frozen, ensure the temp extraction directory is in path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = sys._MEIPASS
    # Add the extraction directory to path
    if application_path not in sys.path:
        sys.path.insert(0, application_path)
    
    # Set environment variables that might help
    os.environ['VI_PHONEME_PATH'] = os.path.join(application_path, 'viphoneme')
    os.environ['VINORM_PATH'] = os.path.join(application_path, 'vinorm')

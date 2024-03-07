from source import utils


class Constants(object):
    # WAKE_THRESHOLD = 0.3  # These values were used for scikit-learn 0.20.3, See:
    # REM_THRESHOLD = 0.35  # https://scikit-learn.org/stable/whats_new.html#version-0-21-0
    WAKE_THRESHOLD = 0.5  #
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = True
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True
    CROPPED_FILE_PATH = utils.get_project_root()+'outputs/cropped/'
    FEATURE_FILE_PATH = utils.get_project_root()+'outputs/features/'
    FIGURE_FILE_PATH = utils.get_project_root()+'outputs/figures/'
    LOWER_BOUND = -0.2
    MATLAB_PATH = 'C://"Program Files"/MATLAB/R2023a/bin/matlab'  # Replace with your MATLAB path

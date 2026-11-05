"""Registry of all detector scripts (used by run_all.py)."""
import importlib

NAMES = ['person_count', 'cap_detection', 'gloves_detection', 'button_detection', 'shoes_detection', 'mask_detection', 'beard_detection', 'object_detection', 'face_recognition']


def get_class(name):
    return importlib.import_module(f"detectors.{name}").DETECTOR

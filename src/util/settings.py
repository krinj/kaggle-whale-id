# -*- coding: utf-8 -*-

"""
Use this to load external or configurable settings from an outside resource (Yaml file).
"""

import os
import shutil
import yaml
from k_util.logger import Logger


class Settings:
    # ===================================================================================================
    # Class methods: Singleton getter.
    # ===================================================================================================

    _instance: 'Settings' = None

    K_SETTINGS_EXAMPLE_FILE = "settings_example.yaml"
    K_SETTINGS_FILE = "settings.yaml"
    K_SETTINGS_ROOT = ".."

    @classmethod
    def instance(cls):
        """ Get the singleton instance if it exists, otherwise create it. """
        if cls._instance is None:
            cls._instance = Settings()
        return cls._instance

    @classmethod
    def get(cls, key: str, default=None):
        """ Get the value of the specified key. If it doesn't exist, get the default. """
        return cls.instance()._get(key, default)

    # ===================================================================================================
    # Instance methods.
    # ===================================================================================================

    def __init__(self):

        # Create the full paths.
        settings_example_path = f"{self.K_SETTINGS_ROOT}/{self.K_SETTINGS_EXAMPLE_FILE}"
        settings_path = f"{self.K_SETTINGS_ROOT}/{self.K_SETTINGS_FILE}"

        # If local settings doesn't exist, then copy it from the example.
        if not os.path.exists(settings_path):
            shutil.copy2(settings_example_path, settings_path)

        # Read the settings yaml.
        with open(settings_path, 'r') as f:
            self.data = yaml.load(f)

        # Print the loaded settings.
        Logger.special("Settings Initialized")
        for k, v in self.data.items():
            Logger.field(k, v)

    def _get(self, key: str, default=None):
        """ Get the value of the specified key. If it doesn't exist, get the default. """
        if key in self.data:
            return self.data[key]
        return default

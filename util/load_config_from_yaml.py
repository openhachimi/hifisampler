import logging
import os
import tempfile
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

logging.basicConfig(format='%(message)s', level=logging.INFO)


def load_config_from_yaml(script_path: Path):
    def decorator(cls):
        try:
            script_dir = script_path.parent
            config_path = script_dir / 'config.yaml'
            default_config_path = script_dir / 'config.default.yaml'

            update_config(config_path, default_config_path)

            active_path = config_path if config_path.exists() else default_config_path
            if active_path.exists():
                logging.info(f"Loading config from: {active_path}")
                yaml = YAML()
                with open(active_path, 'r', encoding='utf-8') as f:
                    nested_config = yaml.load(f) or {}

                for section_name, section_data in nested_config.items():
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            if hasattr(cls, key):
                                setattr(cls, key, value)
                            else:
                                logging.debug(f"Skipping key '{key}' from YAML section '{section_name}'")
            else:
                logging.warning(f"Config file {config_path} not found. Using hardcoded defaults for {cls.__name__}.")

        except Exception as e:
            logging.error(f"Error applying config loader decorator to {cls.__name__}: {e}", exc_info=True)

        logging.info(f"--- Finished applying decorator to {cls.__name__} ---")
        return cls
    return decorator


def update_config(config_path, default_config_path):
    """
    Update the config file to match the structure and version of the default config file,
    while preserving comments, field order, and formatting. Only writes to disk when
    config.yaml does not exist or is missing keys from config.default.yaml.
    """
    config_path = Path(config_path)
    default_config_path = Path(default_config_path)
    if not default_config_path.exists():
        return

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(default_config_path, 'r', encoding='utf-8') as default_file:
        default_config = yaml.load(default_file)

    needs_write = not config_path.exists()
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as config_file:
            current_config = yaml.load(config_file) or CommentedMap()
    else:
        current_config = CommentedMap()

    def recursive_update(current, default):
        nonlocal needs_write
        for key, value in default.items():
            if isinstance(value, dict):
                if key not in current or not isinstance(current[key], dict):
                    needs_write = True
                current[key] = recursive_update(current.get(key, CommentedMap()), value)
            elif isinstance(value, list):
                if key not in current:
                    current[key] = CommentedSeq(value)
                    needs_write = True
                else:
                    current[key] = CommentedSeq(current[key])
            else:
                if key not in current:
                    current[key] = value
                    needs_write = True
        return current

    updated_config = recursive_update(current_config, default_config)

    if not needs_write:
        return

    final_config = CommentedMap()
    for key in default_config:
        final_config[key] = updated_config.pop(key, default_config[key])
    for key, value in updated_config.items():
        final_config[key] = value

    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(config_path.parent), prefix=".config_", suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as config_file:
                yaml.dump(final_config, config_file)
            os.replace(tmp_path, config_path)
            logging.info(f"Config file '{config_path}' updated to version {final_config.get('version')}.")
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    except OSError as e:
        logging.warning(f"Could not write updated config to '{config_path}' (read-only?): {e}")

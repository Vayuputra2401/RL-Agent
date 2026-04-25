import sys
import yaml

def main():
    try:
        with open("openenv.yaml", "r") as f:
            data = yaml.safe_load(f)
        assert "name" in data, "openenv.yaml: Missing 'name'"
        assert "version" in data, "openenv.yaml: Missing 'version'"
        assert "openenv" in data.get("tags", []), "openenv.yaml: Missing 'openenv' tag"
        assert "tasks" in data, "openenv.yaml: Missing 'tasks'"
        print("Validation passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Validation failed: openenv.yaml not found")
        sys.exit(1)
    except Exception as e:
        print(f"Validation failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

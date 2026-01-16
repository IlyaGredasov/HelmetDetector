import os
import shutil
import subprocess
import sys


def copy_git_folder(target: str):
    if os.path.exists(".git"):
        git_dest = os.path.join(target, ".git")
        if os.path.exists(git_dest):
            shutil.rmtree(git_dest)
        shutil.copytree(".git", git_dest, dirs_exist_ok=True)
        print("Copied: .git")

# Скрипт для копирования репозитория на флешку и загрузку в GitLab
def main():
    target = sys.argv[1]
    os.makedirs(target, exist_ok=True)

    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(f"Copying tracked files to {target}")

    for line in result.stdout.splitlines():
        rel_path = line.strip()
        if not rel_path:
            continue

        src = rel_path
        dest = os.path.join(target, rel_path)
        dest_dir = os.path.dirname(dest)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        try:
            shutil.copy2(src, dest)
            print(f"Copied: {rel_path}")
        except FileNotFoundError:
            print(f"ERROR: {rel_path} not found")

    copy_git_folder(target)

    print("Done")


if __name__ == "__main__":
    main()

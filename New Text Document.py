import os

def check_readme_in_folders(base_dir="."):
    for item in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, item)

        if os.path.isdir(folder_path):
            readme_path = os.path.join(folder_path, "README.md")

            if os.path.exists(readme_path):
                print(f"[OK] {item} has README.md")
            else:
                print(f"[MISSING] {item} does NOT have README.md")

if __name__ == "__main__":
    check_readme_in_folders()

    # Keep terminal open until 'q' is pressed
    while True:
        user_input = input("\nPress 'q' then Enter to quit: ").strip().lower()
        if user_input == "q":
            break

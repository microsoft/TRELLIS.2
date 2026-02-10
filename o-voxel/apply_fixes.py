import os

# The list of surgeries we need to perform
tasks = [
    # Fix 1: Explicit casting for IO files (Fixes C2398 Narrowing Conversion Error)
    {
        "path": "src/io/svo.cpp",
        "replacements": [
            ("{svo.size()}", "{(int64_t)svo.size()}"),
            ("{codes.size()}", "{(int64_t)codes.size()}")
        ]
    },
    {
        "path": "src/io/filter_parent.cpp",
        "replacements": [
            ("{N_leaf, C}", "{(int64_t)N_leaf, (int64_t)C}")
        ]
    },
    {
        "path": "src/io/filter_neighbor.cpp",
        "replacements": [
            ("{N, C}", "{(int64_t)N, (int64_t)C}")
        ]
    },
    # Fix 2: Ensure float literals don't trigger truncation warnings/errors
    {
        "path": "src/convert/flexible_dual_grid.cpp",
        "replacements": [
            ("1e-6", "1.0e-6f"), 
            ("= 0.0)", "= 0.0f)") 
        ]
    }
]

print("Applying MSVC Compatibility Fixes...")

for task in tasks:
    file_path = task["path"]
    if not os.path.exists(file_path):
        print(f"WARNING: Could not find {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content
    for old_str, new_str in task["replacements"]:
        # Only replace if the 'new' string isn't already there (idempotency)
        if old_str in new_content and new_str not in new_content:
            new_content = new_content.replace(old_str, new_str)

    if content != new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f" [FIXED] {file_path}")
    else:
        print(f" [CLEAN] {file_path} (No changes needed)")

print("Done. Ready to commit.")
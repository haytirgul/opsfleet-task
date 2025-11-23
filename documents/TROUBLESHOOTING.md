# VS Code Auto-Formatting Troubleshooting Guide

## Issue: Black formatter not running on save

### Quick Fix Steps

#### 1. Reload VS Code Window
**This is the most important step!**
- Press `Ctrl + Shift + P`
- Type: `Developer: Reload Window`
- Press `Enter`

#### 2. Select Python Interpreter
- Press `Ctrl + Shift + P`
- Type: `Python: Select Interpreter`
- Choose **Python 3.11.9** (look for the one that shows 3.11)

#### 3. Check Output Panel
- Press `Ctrl + Shift + U` to open Output panel
- From dropdown, select: **Python**
- Look for any error messages about Black formatter

#### 4. Verify Extension is Active
- Press `Ctrl + Shift + X` to open Extensions
- Search for: `ms-python.black-formatter`
- Should show "Enabled" (not just "Installed")
- Click the extension and check if it needs to be reloaded

#### 5. Manual Test
Open any Python file and try formatting manually:
- Press `Shift + Alt + F` (or `Shift + Option + F` on Mac)
- Or right-click → "Format Document"

If this works but save doesn't, the issue is with the settings.

---

## Common Issues & Solutions

### Issue: "Formatter 'ms-python.black-formatter' is not installed"

**Solution:**
```powershell
code --install-extension ms-python.black-formatter
```
Then reload VS Code window.

---

### Issue: Black is installed but not formatting

**Check 1:** Verify Black is in your Python environment
```bash
py -3.11 -m black --version
```

**Check 2:** Update settings.json
Make sure these lines are in `.vscode/settings.json`:
```json
{
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

**Check 3:** Check for conflicting formatters
Remove these if present:
- `python.formatting.provider`
- Other Python formatter settings

---

### Issue: "Import could not be resolved"

This is a Pylance issue, not a formatter issue.

**Solution:**
- Press `Ctrl + Shift + P`
- Type: `Python: Select Interpreter`
- Make sure Python 3.11 is selected

---

### Issue: Format on save works for other files but not Python

**Solution:**
Check if there's a `.editorconfig` or other config file overriding settings.

---

## Verification Checklist

Run these checks in order:

### ✓ 1. Extensions Installed
```powershell
code --list-extensions | Select-String -Pattern "black-formatter"
```
Should output: `ms-python.black-formatter`

### ✓ 2. Black Available
```bash
py -3.11 -m black --version
```
Should output: `black, 25.11.0 ...`

### ✓ 3. VS Code Settings
Check `.vscode/settings.json` contains:
- `"editor.formatOnSave": true`
- `"[python]": { "editor.defaultFormatter": "ms-python.black-formatter" }`

### ✓ 4. Python Interpreter Selected
Check bottom-left of VS Code status bar:
- Should show: `3.11.9` or similar

### ✓ 5. Manual Format Works
- Open a Python file
- Press `Shift + Alt + F`
- File should be formatted

---

## Still Not Working?

### Nuclear Option: Reset VS Code Settings

1. **Close VS Code completely**

2. **Backup your settings:**
```powershell
Copy-Item .vscode\settings.json .vscode\settings.json.backup
```

3. **Delete workspace settings:**
```powershell
Remove-Item .vscode\settings.json
```

4. **Open VS Code and reinstall extensions:**
```powershell
powershell -ExecutionPolicy Bypass -File install_extensions.ps1
```

5. **Copy back settings:**
```powershell
Copy-Item .vscode\settings.json.backup .vscode\settings.json
```

6. **Reload window:**
`Ctrl + Shift + P` → `Developer: Reload Window`

---

## Test Your Setup

Create a test file to verify formatting:

```python
# test_format.py
def bad_format(x,y,z):
    result=x+y+z
    return result
```

**Save the file** (`Ctrl + S`)

It should automatically format to:
```python
# test_format.py
def bad_format(x, y, z):
    result = x + y + z
    return result
```

If it formats when you save, **formatting is working!** ✅

---

## Get Help

If none of this works:
1. Check VS Code Output panel (`Ctrl + Shift + U`) → Select "Python"
2. Check VS Code Developer Tools (`Help` → `Toggle Developer Tools`) for errors
3. Try opening VS Code from the terminal:
```bash
cd opsfleet-task
code .
```

This ensures VS Code uses the correct Python environment.


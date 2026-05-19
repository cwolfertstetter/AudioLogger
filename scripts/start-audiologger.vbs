' Silent launcher for AudioLogger — no terminal window appears.
' Double-click this file or pin it as a shortcut to your taskbar / Startup folder.
'
' Resolves paths relative to this script, so the file is portable as long as
' it stays at <repo>\scripts\start-audiologger.vbs.

Option Explicit

Dim fso, shell, scriptDir, projectRoot, pythonw, command

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fso.GetParentFolderName(scriptDir)
pythonw = projectRoot & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    MsgBox "AudioLogger venv not found at:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Run 'uv venv' and 'uv pip install -e \"" & ".[gpu,dev]\"' in the project root first.", _
           vbCritical, "AudioLogger"
    WScript.Quit 1
End If

' Run from the project root so that the default 'output_dir: ./recordings' in
' config.yaml resolves to <repo>\recordings.
shell.CurrentDirectory = projectRoot

' Build command: "pythonw.exe" -m audiologger
command = """" & pythonw & """ -m audiologger"

' Run hidden (window style 0), don't wait for completion.
shell.Run command, 0, False

#define MyAppName "PurpleSky"
#define MyAppTitle "PurpleSky launcher - Bybit"
#define MyAppVersion "1.0.0"
#define MyAppExeName "PurpleSky.exe"
#define MyAppPublisher "PurpleSky"

[Setup]
AppId={{5D5F3FA1-2E7E-4B1D-9A11-2D8C3DAE6C79}
AppName={#MyAppTitle}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
LicenseFile=LICENSE
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist\installer
OutputBaseFilename=PurpleSky-Setup
SetupIconFile=purplesky.ico
Compression=lzma
SolidCompression=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\PurpleSky\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  SW_RESTORE = 9;

function SetForegroundWindow(hWnd: HWND): BOOL;
  external 'SetForegroundWindow@user32.dll stdcall';
function ShowWindow(hWnd: HWND; nCmdShow: Integer): BOOL;
  external 'ShowWindow@user32.dll stdcall';

procedure ForceWizardToFront();
begin
  WizardForm.Show;
  ShowWindow(WizardForm.Handle, SW_RESTORE);
  SetForegroundWindow(WizardForm.Handle);
  WizardForm.BringToFront;
end;

procedure InitializeWizard();
begin
  ForceWizardToFront();
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  ForceWizardToFront();
end;

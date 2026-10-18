$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$plugin = Join-Path $repo 'apps\obsidian-plugin'
$manifest = Get-Content (Join-Path $plugin 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.id -ne 'pt-english-tutor') { throw 'Unexpected plugin ID' }
Push-Location $plugin
try {
  npm ci
  if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
  npm run build
  if ($LASTEXITCODE -ne 0) { throw 'Plugin build failed' }
} finally { Pop-Location }
$destination = Join-Path $repo 'release'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$zip = Join-Path $destination ("pt-english-tutor-" + $manifest.version + ".zip")
$files = @('manifest.json','main.js','styles.css') | ForEach-Object { Join-Path $plugin $_ }
foreach ($file in $files) { if (!(Test-Path $file)) { throw "Missing release file $file" } }
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $files -DestinationPath $zip -CompressionLevel Optimal
Add-Type -AssemblyName System.IO.Compression
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
  $entries = @($archive.Entries | ForEach-Object { $_.FullName } | Sort-Object)
  $expected = @('main.js','manifest.json','styles.css')
  if (($entries -join ',') -ne ($expected -join ',')) { throw "Unexpected ZIP entries" }
} finally { $archive.Dispose() }
Write-Output "Verified release package: $zip"

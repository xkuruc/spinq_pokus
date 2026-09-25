param(
    [Parameter(Mandatory = $true)]
    [string]$Token
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$askpass = Join-Path $env:TEMP ("spinq_askpass_{0}.cmd" -f $PID)
$oldAskpass = $env:GIT_ASKPASS
$oldPrompt = $env:GIT_TERMINAL_PROMPT
$oldToken = $env:SPINQ_PUSH_TOKEN

function Check-GitResult {
    if ($LASTEXITCODE -ne 0) {
        throw "Git zlyhal (kod $LASTEXITCODE)."
    }
}

Push-Location $repo
try {
    & git rev-parse --is-inside-work-tree | Out-Null
    Check-GitResult

    # Include source changes and explicitly include measurement output.
    & git add --all
    Check-GitResult
    if (Test-Path -LiteralPath (Join-Path $repo 'results')) {
        & git add --force -- results/
        Check-GitResult
    }

    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        & git -c user.name=xkuruc -c user.email=xkuruc@stuba.sk commit -m "Update SpinQ code and results"
        Check-GitResult
    } elseif ($LASTEXITCODE -ne 0) {
        Check-GitResult
    } else {
        Write-Host 'Ziadne nove subory na commit.'
    }

    # Git obtains the token from this temporary helper. Its file contains
    # only an environment-variable reference, never the token itself.
    Set-Content -LiteralPath $askpass -Value @('@echo off', 'echo %SPINQ_PUSH_TOKEN%') -Encoding Ascii
    $env:SPINQ_PUSH_TOKEN = $Token
    $env:GIT_ASKPASS = $askpass
    $env:GIT_TERMINAL_PROMPT = '0'
    & git -c credential.helper= push https://xkuruc@github.com/xkuruc/spinq_pokus.git HEAD:main
    Check-GitResult
    Write-Host 'Hotovo: commit a push na GitHub.'
} finally {
    $env:GIT_ASKPASS = $oldAskpass
    $env:GIT_TERMINAL_PROMPT = $oldPrompt
    $env:SPINQ_PUSH_TOKEN = $oldToken
    Remove-Item -LiteralPath $askpass -ErrorAction SilentlyContinue
    Pop-Location
}

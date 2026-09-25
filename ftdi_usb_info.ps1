# Read-only FTDI D2XX inventory. It never opens a device or sends data to it.
$ErrorActionPreference = 'Stop'

$dll = Join-Path $env:WINDIR 'System32\ftd2xx.dll'
if (-not (Test-Path -LiteralPath $dll)) {
    Write-Host 'Knižnica ftd2xx.dll nie je dostupná v System32.'
    Write-Host 'USB čip je vo Windows viditeľný, ale priamy prístup cez D2XX zatiaľ nevieme potvrdiť.'
    exit 2
}

$source = @'
using System;
using System.Runtime.InteropServices;

public static class FtdiD2xxInventory {
    [DllImport("ftd2xx.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern UInt32 FT_CreateDeviceInfoList(out UInt32 count);

    [DllImport("ftd2xx.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern UInt32 FT_GetDeviceInfoDetail(
        UInt32 index, out UInt32 flags, out UInt32 type,
        out UInt32 id, out UInt32 location,
        [Out] byte[] serialNumber, [Out] byte[] description,
        out IntPtr handle);
}
'@

try {
    if (-not ([System.Management.Automation.PSTypeName]'FtdiD2xxInventory').Type) {
        Add-Type -TypeDefinition $source
    }
    [uint32]$count = 0
    [uint32]$status = [FtdiD2xxInventory]::FT_CreateDeviceInfoList([ref]$count)
    if ($status -ne 0) { throw "FT_CreateDeviceInfoList vrátil chybu $status" }

    Write-Host "FTDI zariadenia podľa D2XX: $count"
    $found = 0
    $devicesFound = @()
    for ([uint32]$i = 0; $i -lt $count; $i++) {
        [uint32]$flags = 0
        [uint32]$type = 0
        [uint32]$id = 0
        [uint32]$location = 0
        $serial = New-Object byte[] 16
        $description = New-Object byte[] 64
        [IntPtr]$handle = [IntPtr]::Zero
        [uint32]$status = [FtdiD2xxInventory]::FT_GetDeviceInfoDetail(
            $i, [ref]$flags, [ref]$type, [ref]$id, [ref]$location,
            $serial, $description, [ref]$handle)
        if ($status -ne 0) { throw "FT_GetDeviceInfoDetail($i) vrátil chybu $status" }

        if ($id -eq 0x04036014) {
            $found++
            $zero = [Array]::IndexOf($description, [byte]0)
            if ($zero -lt 0) { $zero = $description.Length }
            $name = [Text.Encoding]::ASCII.GetString($description, 0, $zero)
            $chip = if ($type -eq 8) { 'FT232H' } else { "typ $type" }
            $devicesFound += [pscustomobject]@{
                Index = $i
                Chip = $chip
                VID = '0403'
                PID = '6014'
                Description = $name
                OpenedByOtherApp = [bool]($flags -band 1)
            }
        }
    }

    if ($found -eq 0) {
        Write-Host 'D2XX nevidí žiadne zariadenie s VID 0403 a PID 6014.'
    } else {
        $devicesFound | Format-Table -AutoSize | Out-Host
    }
    Write-Host 'Hotovo. Zariadenie nebolo otvorené; neodoslal sa žiadny príkaz.'
} catch {
    Write-Error "Kontrola D2XX zlyhala: $($_.Exception.Message)"
    exit 1
}
